"""
Robustness evaluation for VIPGuard across four image categories:
  1. normal_real  — original real VIP images (r_r_i)
  2. deepfake     — face-swapped and synthesized deepfake images (r_fs_i, r_efs_i)
  3. fgsm_attack  — FGSM adversarial perturbations of real images
  4. pgd_attack   — PGD adversarial perturbations of real images

Metrics computed:
  - Accuracy         = (TP + TN) / N
  - FAR (False Acceptance Rate) = FP / (FP + TN)   [impostors accepted as VIP]
  - FRR (False Rejection Rate)  = FN / (FN + TP)   [real VIPs rejected]
  - Performance Drop %          = (acc_baseline - acc_adv) / acc_baseline * 100
  - Similarity score mean / std per category

Outputs:
  - Console table per VIP identity
  - Aggregate table across all IDs
  - Similarity distribution histograms (optional, --save_plots)
  - JSON results file for programmatic analysis

Usage:
  python evaluate_robustness.py \\
      --ids all \\
      --pretrain_path ./checkpoints/checkpoints_attr_Stage2_merge \\
      --vip_token_dir ./FaceDATA/Pretrained_VIPToken \\
      --adv_root ./data/adversarial \\
      --token_num 32 \\
      --device 0 \\
      --output_dir ./results/before_adv_train \\
      --save_plots
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Constants ───────────────────────────────────────────────────────────────

ALL_IDS = [f"id{k}" for k in range(22)]

CATEGORIES = ['normal_real', 'deepfake', 'fgsm_attack', 'pgd_attack']

# Category display names and expected true label (1=VIP, 0=non-VIP/impostor)
CATEGORY_META = {
    'normal_real': {'label': 1, 'display': 'Normal Real'},
    'deepfake':    {'label': 0, 'display': 'Deepfake'},
    'fgsm_attack': {'label': 1, 'display': 'FGSM Attack'},
    'pgd_attack':  {'label': 1, 'display': 'PGD Attack'},
}


# ─── Model Loading ────────────────────────────────────────────────────────────

from attacks.utils import load_face_fg, load_center, sim_from_path, load_vip_token_into


def load_vipguard_base(
    pretrain_path: str,
    token_num: int,
    device_map: str = 'auto',
    load_in_4bit: bool = True,
) -> Tuple:
    """
    Load the VLM once without any VIP token.
    Call load_vip_token_into(model, path) before each identity's evaluation
    to swap the active VIP token without reloading the 15.6GB model.

    load_in_4bit=True uses bitsandbytes NF4 quantization to reduce the 14GB
    model to ~3.5GB, fitting it entirely on a 12GB GPU and avoiding disk
    offloading which makes inference ~48 min/image.
    """
    from Models.VIPGuard import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from Models.Wrapper import Wrapper

    print(f"Loading VLM from {pretrain_path} (once for all VIPs)...")

    load_kwargs = dict(
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs['quantization_config'] = bnb_config
            print("  Using 4-bit NF4 quantization (bitsandbytes) to fit model on GPU.")
        except ImportError:
            print("  [WARN] bitsandbytes not available; falling back to float16.")
            load_kwargs['torch_dtype'] = torch.float16
    else:
        load_kwargs['torch_dtype'] = torch.float16

    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrain_path, **load_kwargs
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(pretrain_path)
    model = Wrapper(
        vl_model=vl_model,
        processor=processor,
        optim_facechecker=False,
        vip_token_num=token_num,
    )
    # Don't call model.half() when using 4-bit — quantized weights must stay in
    # bnb format. Instead, cast only the non-quantized modules (FaceChecker, VIP
    # token) to float16 to match the model's compute dtype.
    if load_in_4bit:
        model.vl_model.facechecker.half()
    else:
        model = model.half()
    print("VLM loaded.")
    return model, processor


# ─── Similarity ──────────────────────────────────────────────────────────────

def get_similarity(img_path: str, fg_face, center: torch.Tensor) -> int:
    """Compute similarity score (0–100) for an image file."""
    return sim_from_path(img_path, fg_face, center)


# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(
    img_path: str,
    sim_score: int,
    model,
    processor,
    token_num: int,
    max_image_size: int = 224,
) -> str:
    """
    Run VIPGuard inference on a single image.

    Returns the raw model output string containing <Conclusion>[Yes/No] ...
    """
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    # Prompt format MUST match Stage3 training data exactly:
    # - Image comes first in content
    # - Text has two <|face_pad|> tokens: one at start, one after "face tokens are shown"
    # This produces 2×face_len face token positions in input_ids, matching the 2× FaceChecker path.
    prompt = (
        f"<|face_pad|>Please determine whether the person in the input image is VIP user. "
        f"The face similarity between the input face and VIP user is {sim_score}/100. "
        f"The face tokens are shown as follows, <|face_pad|> . "
        f"You should first give your answer by 'yes' or 'no'. "
        f"Then, you should explain your reasoning step by step based on different facial attributes."
    )
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": prompt},
        ]
    }]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)

    # Resize if needed (memory-safe)
    for i in range(len(image_inputs)):
        h, w = image_inputs[i].size
        if max(h, w) > max_image_size:
            scale = max_image_size / max(h, w)
            image_inputs[i] = image_inputs[i].resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

    face_len = model.vl_model.facechecker.face_checker.vip_prompt.data.shape[0]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        our_token=True,
        our_token_length=token_num,
        face_pad=True,
        face_length=face_len,
    )
    inputs = {
        k: v.to("cuda").to(torch.float16) if v.dtype == torch.float32 else v.to("cuda")
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40)

    return processor.batch_decode(output, skip_special_tokens=True)[0]


def parse_prediction(output_text: str) -> Optional[int]:
    """
    Parse Yes/No prediction from VIPGuard output.

    Returns 1 for 'Yes' (authenticated as VIP), 0 for 'No', None if unparseable.
    """
    # Match <Conclusion>[Yes] or <Conclusion>[No] (case-insensitive)
    match = re.search(r'<Conclusion>\[(Yes|No)\]', output_text, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == 'yes' else 0
    # Fallback: look for standalone yes/no after "assistant"
    lines = output_text.lower().split('\n')
    for line in reversed(lines):
        if 'yes' in line:
            return 1
        if 'no' in line:
            return 0
    return None


# ─── Evaluation Set Building ─────────────────────────────────────────────────

def build_evaluation_entries(
    id_name: str,
    data_root: str,
    adv_root: str,
    fg_face,
    center: torch.Tensor,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Enumerate evaluation images for all 4 categories for one VIP.

    Returns dict mapping category name to list of (img_path, sim_score, true_label).
      true_label: 1 = authentic VIP, 0 = impostor / deepfake
    """
    entries = {cat: [] for cat in CATEGORIES}

    # 1. normal_real: r_r_i/*/2.png (true label = 1)
    real_dir = os.path.join(data_root, id_name, 'r_r_i')
    if os.path.isdir(real_dir):
        for pair in sorted(os.listdir(real_dir)):
            for candidate in ['2.png', 'fake.png', 'probe.png']:
                p = os.path.join(real_dir, pair, candidate)
                if os.path.isfile(p):
                    sim = get_similarity(p, fg_face, center)
                    entries['normal_real'].append((p, sim, 1))
                    break

    # 2. deepfake: r_fs_i and r_efs_i (true label = 0)
    for fake_type in ['r_fs_i', 'r_efs_i']:
        fake_dir = os.path.join(data_root, id_name, fake_type)
        if os.path.isdir(fake_dir):
            for pair in sorted(os.listdir(fake_dir)):
                for candidate in ['fake.png', '2.png']:
                    p = os.path.join(fake_dir, pair, candidate)
                    if os.path.isfile(p):
                        sim = get_similarity(p, fg_face, center)
                        entries['deepfake'].append((p, sim, 0))
                        break

    # 3. fgsm_attack: adversarial real images (true label = 1)
    fgsm_dir = os.path.join(adv_root, 'fgsm', 'adv_real', id_name)
    if os.path.isdir(fgsm_dir):
        for pair in sorted(os.listdir(fgsm_dir)):
            p = os.path.join(fgsm_dir, pair, 'adv.png')
            if os.path.isfile(p):
                sim_txt = os.path.join(fgsm_dir, pair, 'sim.txt')
                if os.path.isfile(sim_txt):
                    sim = int(open(sim_txt).read().strip())
                else:
                    sim = get_similarity(p, fg_face, center)
                entries['fgsm_attack'].append((p, sim, 1))

    # 4. pgd_attack: adversarial real images (true label = 1)
    pgd_dir = os.path.join(adv_root, 'pgd', 'adv_real', id_name)
    if os.path.isdir(pgd_dir):
        for pair in sorted(os.listdir(pgd_dir)):
            p = os.path.join(pgd_dir, pair, 'adv.png')
            if os.path.isfile(p):
                sim_txt = os.path.join(pgd_dir, pair, 'sim.txt')
                if os.path.isfile(sim_txt):
                    sim = int(open(sim_txt).read().strip())
                else:
                    sim = get_similarity(p, fg_face, center)
                entries['pgd_attack'].append((p, sim, 1))

    return entries


# ─── Per-Category Evaluation ─────────────────────────────────────────────────

def evaluate_category(
    entries: List[Tuple[str, int, int]],
    model,
    processor,
    token_num: int,
    category_name: str,
    max_images: Optional[int] = None,
) -> dict:
    """
    Run VIPGuard inference on all images in one category and compute metrics.

    Returns dict with:
      accuracy, FAR, FRR, n_total, n_correct, sim_scores, predictions, labels
    """
    if not entries:
        return {
            'accuracy': None, 'FAR': None, 'FRR': None,
            'n_total': 0, 'n_correct': 0,
            'sim_scores': [], 'predictions': [], 'labels': [],
        }

    if max_images is not None:
        entries = entries[:max_images]

    predictions = []
    labels = []
    sim_scores = []

    for i, (img_path, sim, true_label) in enumerate(entries):
        print(f"    [{category_name}] {i+1}/{len(entries)}: {os.path.basename(img_path)}", end=' ')
        try:
            output = run_inference(img_path, sim, model, processor, token_num)
            pred = parse_prediction(output)
        except Exception as e:
            print(f"[ERROR: {e}]")
            pred = None

        if pred is None:
            print("[UNPARSEABLE]")
            pred = 0  # conservative: treat unparseable as rejection

        match_str = 'CORRECT' if pred == true_label else 'WRONG'
        print(f"pred={pred} true={true_label} sim={sim} [{match_str}]")

        predictions.append(pred)
        labels.append(true_label)
        sim_scores.append(sim)

    # Compute confusion matrix elements
    TP = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    TN = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    FP = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    FN = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    N = len(predictions)

    accuracy = (TP + TN) / N if N > 0 else None
    FAR = FP / (FP + TN) if (FP + TN) > 0 else None   # impostors accepted
    FRR = FN / (FN + TP) if (FN + TP) > 0 else None   # real VIPs rejected

    return {
        'accuracy': accuracy,
        'FAR': FAR,
        'FRR': FRR,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'n_total': N,
        'n_correct': TP + TN,
        'sim_scores': sim_scores,
        'sim_mean': float(np.mean(sim_scores)) if sim_scores else None,
        'sim_std': float(np.std(sim_scores)) if sim_scores else None,
        'predictions': predictions,
        'labels': labels,
    }


# ─── Results Display ──────────────────────────────────────────────────────────

def format_pct(val: Optional[float], suffix: str = '%') -> str:
    if val is None:
        return '  —  '
    return f"{val*100:.1f}{suffix}"


def print_metrics_table(results: Dict[str, dict], id_name: str):
    """Print a formatted metrics table for one VIP identity."""
    baseline_acc = results.get('normal_real', {}).get('accuracy')

    print(f"\n{'='*72}")
    print(f"  VIP: {id_name:10s}  |  Adversarial Robustness Evaluation")
    print(f"{'='*72}")
    print(f"  {'Category':<16} {'Acc':>7} {'FAR':>7} {'FRR':>7} {'SimMean':>9} {'N':>5}  {'Note'}")
    print(f"  {'-'*68}")

    for cat in CATEGORIES:
        r = results.get(cat)
        if not r or r['n_total'] == 0:
            print(f"  {CATEGORY_META[cat]['display']:<16} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>9} {'0':>5}")
            continue

        acc_str = format_pct(r['accuracy'])
        far_str = format_pct(r['FAR']) if r['FAR'] is not None else '  —  '
        frr_str = format_pct(r['FRR']) if r['FRR'] is not None else '  —  '
        sim_str = f"{r['sim_mean']:.1f}" if r['sim_mean'] is not None else '  —  '

        note = ''
        if cat in ('fgsm_attack', 'pgd_attack') and baseline_acc and r['accuracy'] is not None:
            drop = (baseline_acc - r['accuracy']) / baseline_acc * 100
            note = f"[DROP: -{drop:.1f}%]"

        print(f"  {CATEGORY_META[cat]['display']:<16} {acc_str:>7} {far_str:>7} {frr_str:>7} {sim_str:>9} {r['n_total']:>5}  {note}")

    print(f"  {'-'*68}")


def print_aggregate_table(all_results: Dict[str, Dict[str, dict]]):
    """Print aggregate metrics across all VIP identities."""
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS ({len(all_results)} VIP identities)")
    print(f"{'='*72}")
    print(f"  {'Category':<16} {'AvgAcc':>8} {'AvgFAR':>8} {'AvgFRR':>8} {'AvgSim':>9}")
    print(f"  {'-'*60}")

    for cat in CATEGORIES:
        accs, fars, frrs, sims = [], [], [], []
        for results in all_results.values():
            r = results.get(cat, {})
            if r and r['n_total'] > 0:
                if r['accuracy'] is not None: accs.append(r['accuracy'])
                if r['FAR'] is not None: fars.append(r['FAR'])
                if r['FRR'] is not None: frrs.append(r['FRR'])
                if r['sim_mean'] is not None: sims.append(r['sim_mean'])

        avg_acc = f"{np.mean(accs)*100:.1f}%" if accs else 'N/A'
        avg_far = f"{np.mean(fars)*100:.1f}%" if fars else '—'
        avg_frr = f"{np.mean(frrs)*100:.1f}%" if frrs else '—'
        avg_sim = f"{np.mean(sims):.1f}" if sims else 'N/A'

        print(f"  {CATEGORY_META[cat]['display']:<16} {avg_acc:>8} {avg_far:>8} {avg_frr:>8} {avg_sim:>9}")

    print(f"  {'-'*60}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_similarity_distributions(
    results: Dict[str, dict],
    id_name: str,
    output_dir: str,
):
    """
    Plot overlaid similarity score histograms for all 4 categories.

    Saves to: output_dir/{id_name}_sim_distribution.png
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'normal_real': '#2196F3', 'deepfake': '#F44336',
              'fgsm_attack': '#FF9800', 'pgd_attack': '#9C27B0'}

    plotted = False
    for cat in CATEGORIES:
        r = results.get(cat)
        if not r or not r['sim_scores']:
            continue
        ax.hist(r['sim_scores'], bins=20, range=(0, 100), alpha=0.5,
                color=colors[cat], label=f"{CATEGORY_META[cat]['display']} (n={r['n_total']})",
                edgecolor='white', linewidth=0.5)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel('Similarity Score (0–100)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'VIPGuard Similarity Score Distribution — VIP: {id_name}', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{id_name}_sim_distribution.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VIPGuard adversarial robustness across 4 image categories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ids', type=str, default='all',
                        help='Comma-separated VIP IDs or "all"')
    parser.add_argument('--pretrain_path', type=str,
                        default='./checkpoints/checkpoints_attr_Stage2_merge',
                        help='Path to Stage2 pretrained checkpoint')
    parser.add_argument('--vip_token_dir', type=str,
                        default='./FaceDATA/Pretrained_VIPToken',
                        help='Directory containing per-VIP vip_token.pt files')
    parser.add_argument('--adv_root', type=str, default='./data/adversarial',
                        help='Root directory containing adversarial images')
    parser.add_argument('--data_root', type=str, default='./FaceDATA/Training_Img',
                        help='Root directory for training images')
    parser.add_argument('--center_dir', type=str, default='./FaceDATA/FaceEmb_Center',
                        help='Directory containing VIP embedding centers')
    parser.add_argument('--token_num', type=int, default=32,
                        help='Number of VIP tokens used during training')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device index or "cpu"')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for saving results (JSON + plots)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save similarity distribution plots')
    parser.add_argument('--categories', type=str, default='all',
                        help='Comma-separated categories to evaluate, or "all"')
    parser.add_argument('--max_per_category', type=int, default=None,
                        help='Maximum images to evaluate per category (None = all). '
                             'Use e.g. 50 for a fast representative sample.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.device.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Resolve IDs
    if args.ids.lower() == 'all':
        ids_to_eval = ALL_IDS
    else:
        ids_to_eval = [x.strip() for x in args.ids.split(',')]

    # Resolve categories
    if args.categories.lower() == 'all':
        cats_to_eval = CATEGORIES
    else:
        cats_to_eval = [x.strip() for x in args.categories.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load TransFace once — used only for similarity scoring, stays on CPU
    print("Loading TransFace for similarity computation...")
    fg_face = load_face_fg('cpu')
    print("TransFace loaded.")

    # Load VLM once — only the tiny VIP token tensor is swapped per identity,
    # avoiding 22 full 15.6GB model loads.
    model, processor = load_vipguard_base(
        pretrain_path=args.pretrain_path,
        token_num=args.token_num,
    )

    all_results = {}

    for id_name in ids_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {id_name}")
        print(f"{'='*60}")

        center_path = os.path.join(args.center_dir, f'{id_name}.ckpt')
        vip_token_path = os.path.join(args.vip_token_dir, f'{id_name}.pt')
        if not os.path.isfile(vip_token_path):
            vip_token_path = os.path.join(args.vip_token_dir, id_name, 'vip_token.pt')

        if not os.path.isfile(center_path):
            print(f"  [SKIP] Center not found: {center_path}")
            continue
        if not os.path.isfile(vip_token_path):
            print(f"  [SKIP] VIP token not found: {vip_token_path}")
            continue

        center = load_center(id_name, args.center_dir)

        # Swap just the VIP token — no model reload needed
        print(f"  Swapping VIP token from {vip_token_path}")
        load_vip_token_into(model, vip_token_path)

        print("  Building evaluation image lists...")
        all_entries = build_evaluation_entries(
            id_name=id_name,
            data_root=args.data_root,
            adv_root=args.adv_root,
            fg_face=fg_face,
            center=center,
        )

        id_results = {}
        for cat in cats_to_eval:
            entries = all_entries.get(cat, [])
            n_eval = min(len(entries), args.max_per_category) if args.max_per_category else len(entries)
            print(f"\n  Category: {cat} ({len(entries)} total, evaluating {n_eval})")
            if not entries:
                print(f"  [SKIP] No images found for {cat}")
                id_results[cat] = {'n_total': 0}
                continue
            id_results[cat] = evaluate_category(
                entries=entries,
                model=model,
                processor=processor,
                token_num=args.token_num,
                category_name=cat,
                max_images=args.max_per_category,
            )

        all_results[id_name] = id_results
        print_metrics_table(id_results, id_name)

        if args.save_plots:
            plot_similarity_distributions(id_results, id_name, args.output_dir)

        torch.cuda.empty_cache()

    # Aggregate table
    if len(all_results) > 1:
        print_aggregate_table(all_results)

    # Save JSON results
    results_path = os.path.join(args.output_dir, 'robustness_results.json')
    # Convert results to JSON-serializable format (remove non-serializable items)
    serializable = {}
    for id_name, id_res in all_results.items():
        serializable[id_name] = {}
        for cat, r in id_res.items():
            if not r:
                continue
            serializable[id_name][cat] = {
                k: v for k, v in r.items()
                if k not in ('predictions', 'labels', 'sim_scores')
            }
            if 'sim_scores' in r:
                serializable[id_name][cat]['sim_scores'] = r['sim_scores']

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
