"""
Generate adversarial images for all 22 VIP identities using FGSM and PGD attacks.

This script attacks TransFace face embeddings to produce images that:
  - Look visually identical to the original (imperceptible perturbations)
  - Have significantly lower cosine similarity to the VIP center embedding
  - Can be used to evaluate and improve model robustness

Output structure:
  data/adversarial/fgsm/adv_real/id{k}/{pair_name}/adv.png   (adversarial real images)
  data/adversarial/fgsm/adv_real/id{k}/{pair_name}/sim.txt   (new similarity score)
  data/adversarial/pgd/adv_real/id{k}/{pair_name}/adv.png
  data/adversarial/pgd/adv_real/id{k}/{pair_name}/sim.txt
  data/adversarial/json/id{k}_adv_mixed.json                  (mixed training JSON)

The 'adv_real' substring in output paths is required so that VIP_Dataset.py assigns
label=0 (authentic) to these images — the model should learn to still recognize
the real VIP despite adversarial perturbations.

Usage:
  python generate_adversarial_dataset.py --ids all --attack both --device 0
  python generate_adversarial_dataset.py --ids id0,id1 --attack fgsm --epsilon 0.03137
"""

import os
import sys
import json
import argparse
import random

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.fgsm import fgsm_attack, preprocess_for_attack
from attacks.pgd import pgd_attack
from attacks.utils import load_face_backbone, load_center, sim_from_array


# ─── Constants ──────────────────────────────────────────────────────────────

ALL_IDS = [f"id{k}" for k in range(22)]

# Prompt format MUST match evaluate_robustness.py exactly so training and evaluation
# see the same input distribution. Using the short "just say yes/no" format caused
# near-zero training loss (no gradient), because the model trivially answered correctly
# without having to reason through the low similarity score.
# With the long format, the model must generate a full attribute-based justification
# for why a low-sim image is still the VIP — this produces high initial loss on
# adversarial examples, which drives the VIP token updates.
ADV_REAL_PROMPT = (
    "<|face_pad|><image>Please determine whether the person in the input image is VIP user. "
    "The face similarity between the input face and VIP user is {sim}/100. "
    "The face tokens are shown as follows, <|face_pad|> . "
    "You should first give your answer by 'yes' or 'no'. "
    "Then, you should explain your reasoning step by step based on different facial attributes."
)
ADV_REAL_ANSWER = (
    "<Conclusion>[Yes] The two images are of the same person.</Conclusion>\n"
    "<Analysis>\n"
    "The face similarity score is reduced by adversarial perturbation, but the underlying "
    "facial structure matches the registered VIP profile.\n"
    "1. Outer face region: [No] The face shape and jawline are consistent with the VIP.\n"
    "2. Eye region: [No] Eye spacing and shape match the VIP profile.\n"
    "3. Nose region: [No] Nose structure is consistent.\n"
    "4. Mouth region: [No] Mouth shape matches.\n"
    "5. Skin region: [No] Skin tone is consistent.\n"
    "Overall Conclusion:\n"
    "Despite the low similarity score caused by image perturbation, the facial attributes "
    "confirm this is the registered VIP.\n"
    "</Analysis>"
)


# ─── Image Enumeration ───────────────────────────────────────────────────────

def get_source_images(id_name: str, data_root: str = './FaceDATA/Training_Img'):
    """
    Enumerate all real same-person probe images for one VIP identity.

    Only r_r_i (Real vs Real, same identity) images are used as the attack source —
    these are the images an adversary would try to craft to evade detection.

    Returns: list of (img_path, pair_name) tuples.
      img_path:  path to the probe image (2.png or real.png patterns)
      pair_name: subfolder name, e.g. 'id0_38'
    """
    results = []
    base = os.path.join(data_root, id_name, 'r_r_i')
    if not os.path.isdir(base):
        return results

    for pair_dir in sorted(os.listdir(base)):
        pair_path = os.path.join(base, pair_dir)
        if not os.path.isdir(pair_path):
            continue
        # Try common probe image names
        for candidate in ['2.png', 'fake.png', 'probe.png']:
            img_path = os.path.join(pair_path, candidate)
            if os.path.isfile(img_path):
                results.append((img_path, pair_dir))
                break

    return results


# ─── Core Generation ─────────────────────────────────────────────────────────

def generate_adv_for_id(
    id_name: str,
    fg_face,
    center: torch.Tensor,
    attack_type: str,
    output_root: str = './data/adversarial',
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    num_steps: int = 10,
    device: str = 'cuda',
    skip_existing: bool = False,
):
    """
    Generate adversarial images for all source images of one VIP identity.

    Saves to: output_root/{attack_type}/adv_real/{id_name}/{pair_name}/adv.png
              output_root/{attack_type}/adv_real/{id_name}/{pair_name}/sim.txt

    Returns: list of JSON-ready dicts for building mixed training files.
    """
    backbone = fg_face.face_model  # raw ViT needed so gradients flow during attacks
    source_images = get_source_images(id_name)
    if not source_images:
        print(f"  [WARN] No source images found for {id_name}")
        return []

    out_dir = os.path.join(output_root, attack_type, 'adv_real', id_name)
    os.makedirs(out_dir, exist_ok=True)

    json_entries = []

    for img_path, pair_name in source_images:
        pair_out = os.path.join(out_dir, pair_name)
        os.makedirs(pair_out, exist_ok=True)
        adv_img_path = os.path.join(pair_out, 'adv.png')
        sim_txt_path  = os.path.join(pair_out, 'sim.txt')
        orig_sim_path = os.path.join(pair_out, 'orig_sim.txt')

        # Skip if already generated — reload both sims from disk
        if skip_existing and os.path.isfile(adv_img_path):
            adv_sim  = int(open(sim_txt_path).read().strip())  if os.path.isfile(sim_txt_path)  else -1
            orig_sim = int(open(orig_sim_path).read().strip()) if os.path.isfile(orig_sim_path) else adv_sim
            rel_path = adv_img_path.replace('\\', '/')
            if not rel_path.startswith('./'):
                rel_path = './' + rel_path.lstrip('/')
            json_entries.append({'img_path': rel_path, 'adv_sim': adv_sim, 'orig_sim': orig_sim})
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [WARN] Cannot read {img_path}, skipping.")
            continue

        # Original (pre-attack) similarity — used in training prompts so there is no
        # conflict between the text signal ("similarity 85/100") and the visual label
        # ([Yes] this is the VIP). The attack lowers the computed score, but the TRUE
        # identity similarity is the pre-attack value.
        orig_sim = sim_from_array(img_bgr, fg_face, center)

        # Run attack
        try:
            if attack_type == 'fgsm':
                adv_bgr = fgsm_attack(backbone, img_bgr, center, epsilon=epsilon, device=device)
            else:
                adv_bgr = pgd_attack(backbone, img_bgr, center, epsilon=epsilon,
                                     alpha=alpha, num_steps=num_steps, device=device)
        except Exception as e:
            print(f"  [ERROR] Attack failed on {img_path}: {e}")
            continue

        # Adversarial (post-attack) similarity — used in evaluate_robustness.py (real-world eval)
        adv_sim = sim_from_array(adv_bgr, fg_face, center)

        # Save outputs
        cv2.imwrite(adv_img_path, adv_bgr)
        with open(sim_txt_path, 'w') as f:
            f.write(str(adv_sim))
        with open(orig_sim_path, 'w') as f:
            f.write(str(orig_sim))

        # Normalize path for JSON (relative, forward slashes)
        rel_path = os.path.relpath(adv_img_path).replace('\\', '/')
        if not rel_path.startswith('./'):
            rel_path = './' + rel_path
        json_entries.append({'img_path': rel_path, 'adv_sim': adv_sim, 'orig_sim': orig_sim})

    print(f"  [{attack_type.upper()}] {id_name}: {len(json_entries)} adversarial images generated")
    return json_entries


# ─── JSON Construction ────────────────────────────────────────────────────────

def build_adversarial_json_entries(adv_entries):
    """
    Convert attack result dicts to Stage3 training JSON format.

    Uses the ADVERSARIAL (post-attack) similarity score in the prompt, matching
    the distribution seen at evaluation time (evaluate_robustness.py also uses
    adv_sim from sim.txt). Training with orig_sim caused a train/eval mismatch:
    the model learned "high sim + VIP tokens → Yes" but at eval saw low sim and
    always said No. Training with adv_sim forces the model to learn to say Yes
    even when the text says 47–75/100, by relying on VIP token embeddings.
    """
    records = []
    for entry in adv_entries:
        # Use adv_sim to match eval distribution exactly
        train_sim = entry.get('adv_sim', entry.get('sim', 50))
        train_sim = max(0, min(100, train_sim))
        prompt = ADV_REAL_PROMPT.format(sim=train_sim)
        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ADV_REAL_ANSWER}
            ],
            "images": [entry['img_path']]
        }
        records.append(record)
    return records


def build_mixed_json(
    id_name: str,
    fgsm_entries,
    pgd_entries,
    original_json_path: str,
    output_path: str,
    adv_fraction: float = 0.25,
    seed: int = 42,
):
    """
    Build a mixed training JSON combining original Stage3 data with adversarial samples.

    Adversarial samples comprise adv_fraction * len(original) entries total
    (split equally between FGSM and PGD if both provided).

    Args:
        id_name: VIP identity name (for logging).
        fgsm_entries: JSON records for FGSM adversarial images.
        pgd_entries:  JSON records for PGD adversarial images.
        original_json_path: Path to the original Stage3 training JSON.
        output_path: Path to write the mixed JSON.
        adv_fraction: Fraction of adversarial samples relative to original count.
        seed: Random seed for reproducible shuffling.
    """
    if not os.path.isfile(original_json_path):
        print(f"  [WARN] Original JSON not found: {original_json_path}. Skipping mix.")
        return

    with open(original_json_path, 'r', encoding='utf-8') as f:
        original = json.load(f)

    n_original = len(original)
    n_adv_total = max(1, int(n_original * adv_fraction))

    # Split budget between FGSM and PGD
    all_adv = []
    if fgsm_entries and pgd_entries:
        n_each = n_adv_total // 2
        all_adv = fgsm_entries[:n_each] + pgd_entries[:n_each]
    elif fgsm_entries:
        all_adv = fgsm_entries[:n_adv_total]
    elif pgd_entries:
        all_adv = pgd_entries[:n_adv_total]

    combined = original + all_adv
    rng = random.Random(seed)
    rng.shuffle(combined)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"  [JSON] {id_name}: {n_original} original + {len(all_adv)} adv = {len(combined)} total -> {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FGSM and PGD adversarial images for VIPGuard robustness evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ids', type=str, default='all',
                        help='Comma-separated list of VIP IDs (e.g. id0,id1) or "all"')
    parser.add_argument('--attack', type=str, default='both', choices=['fgsm', 'pgd', 'both'],
                        help='Which attack(s) to generate')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device index (e.g. 0) or "cpu"')
    parser.add_argument('--epsilon', type=float, default=8 / 255,
                        help='Perturbation budget in pixel-space [0,1] (default: 8/255 ≈ 0.0314)')
    parser.add_argument('--alpha', type=float, default=2 / 255,
                        help='PGD step size in pixel-space (default: 2/255 ≈ 0.00784)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of PGD iterations')
    parser.add_argument('--output_root', type=str, default='./data/adversarial',
                        help='Root directory for saving adversarial images and JSONs')
    parser.add_argument('--data_root', type=str, default='./FaceDATA/Training_Img',
                        help='Root directory for VIP training images')
    parser.add_argument('--center_dir', type=str, default='./FaceDATA/FaceEmb_Center',
                        help='Directory containing VIP embedding center files')
    parser.add_argument('--json_dir', type=str, default='./FaceDATA/Stage3_training_json',
                        help='Directory containing original Stage3 training JSONs')
    parser.add_argument('--adv_fraction', type=float, default=0.25,
                        help='Fraction of adversarial samples in mixed training JSON')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip image pairs that have already been generated')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.device.lower() == 'cpu':
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Resolve VIP IDs to process
    if args.ids.lower() == 'all':
        ids_to_process = ALL_IDS
    else:
        ids_to_process = [x.strip() for x in args.ids.split(',')]
    print(f"Processing {len(ids_to_process)} VIP identities: {ids_to_process}")

    # Determine which attacks to run
    run_fgsm = args.attack in ('fgsm', 'both')
    run_pgd = args.attack in ('pgd', 'both')

    # Load FG_Face once — fg_face.face_model is the raw ViT for attack gradients,
    # fg_face.face_emb_forward is used for @no_grad similarity scoring post-attack.
    print("Loading TransFace...")
    from attacks.utils import load_face_fg
    fg_face = load_face_fg(device)
    print("TransFace loaded.")

    json_out_dir = os.path.join(args.output_root, 'json')
    os.makedirs(json_out_dir, exist_ok=True)

    for id_name in ids_to_process:
        print(f"\n{'='*50}")
        print(f"Processing: {id_name}")
        print(f"{'='*50}")

        # Load VIP embedding center
        try:
            center = load_center(id_name, args.center_dir)
        except FileNotFoundError:
            print(f"  [SKIP] Center file not found for {id_name}")
            continue

        fgsm_json_entries = []
        pgd_json_entries = []

        if run_fgsm:
            print(f"  Running FGSM (epsilon={args.epsilon:.4f})...")
            raw_entries = generate_adv_for_id(
                id_name=id_name,
                fg_face=fg_face,
                center=center,
                attack_type='fgsm',
                output_root=args.output_root,
                epsilon=args.epsilon,
                device=device,
                skip_existing=args.skip_existing,
            )
            fgsm_json_entries = build_adversarial_json_entries(raw_entries)

        if run_pgd:
            print(f"  Running PGD (epsilon={args.epsilon:.4f}, alpha={args.alpha:.4f}, steps={args.steps})...")
            raw_entries = generate_adv_for_id(
                id_name=id_name,
                fg_face=fg_face,
                center=center,
                attack_type='pgd',
                output_root=args.output_root,
                epsilon=args.epsilon,
                alpha=args.alpha,
                num_steps=args.steps,
                device=device,
                skip_existing=args.skip_existing,
            )
            pgd_json_entries = build_adversarial_json_entries(raw_entries)

        # Build mixed training JSON
        original_json = os.path.join(args.json_dir, f'{id_name}.json')
        mixed_json_out = os.path.join(json_out_dir, f'{id_name}_adv_mixed.json')
        build_mixed_json(
            id_name=id_name,
            fgsm_entries=fgsm_json_entries,
            pgd_entries=pgd_json_entries,
            original_json_path=original_json,
            output_path=mixed_json_out,
            adv_fraction=args.adv_fraction,
        )

    print(f"\nDone. Adversarial data saved to: {args.output_root}")


if __name__ == '__main__':
    main()
