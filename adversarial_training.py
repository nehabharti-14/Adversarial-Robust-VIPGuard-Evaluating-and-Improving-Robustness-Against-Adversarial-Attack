"""
Adversarially fine-tune VIP tokens on a mixed dataset (original + adversarial images).

This script reuses Stage3_train.train_one_epoch() directly — no changes to that
function are needed. Only VIP token embeddings are updated (the 32B LLM stays frozen).

Strategy:
  - Mix original Stage3 training data with adversarial images (FGSM + PGD)
  - Adversarial images are labeled 'Yes' (authentic): the model should learn to
    still authenticate real VIP images despite adversarial noise
  - Use a lower learning rate (0.01 vs original 1.0) since tokens are already trained
  - Mixed JSONs produced by generate_adversarial_dataset.py are used directly

Output checkpoints:
  ./checkpoints/Stage3_adv/{id_name}/vip_token.pt

Usage:
  python adversarial_training.py \\
      --ids all \\
      --pretrain_path ./checkpoints/checkpoints_attr_Stage2_merge \\
      --vip_token_dir ./FaceDATA/Pretrained_VIPToken \\
      --adv_root ./data/adversarial \\
      --token_num 32 \\
      --epochs 2 \\
      --lr 0.01 \\
      --device 0 \\
      --use_mixed_precision \\
      --mixed_precision_dtype bf16
"""

import os
import sys
import json
import random
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.utils import load_vip_token_into


# ─── Constants ────────────────────────────────────────────────────────────────

ALL_IDS = [f"id{k}" for k in range(22)]


# ─── JSON Utilities ───────────────────────────────────────────────────────────

def build_mixed_json_if_missing(
    id_name: str,
    adv_root: str,
    json_dir: str,
    output_path: str,
    adv_fraction: float = 0.25,
    seed: int = 42,
) -> bool:
    """
    Build a mixed training JSON if one does not exist yet.

    Combines original Stage3 training data with adversarial samples from
    data/adversarial/json/{id_name}_adv_mixed.json (produced by
    generate_adversarial_dataset.py), or falls back to building from scratch.

    Returns True if a usable JSON was found or created, False otherwise.
    """
    # If a pre-built mixed JSON already exists (from generate_adversarial_dataset.py), use it
    adv_mixed = os.path.join(adv_root, 'json', f'{id_name}_adv_mixed.json')
    if os.path.isfile(adv_mixed):
        print(f"  Using pre-built mixed JSON: {adv_mixed}")
        return True  # caller uses adv_mixed path directly

    # Otherwise, try to build one from components
    original_json = os.path.join(json_dir, f'{id_name}.json')
    if not os.path.isfile(original_json):
        print(f"  [SKIP] Original JSON not found: {original_json}")
        return False

    with open(original_json, 'r', encoding='utf-8') as f:
        original = json.load(f)

    # Collect any available adversarial entries from individual attack JSONs
    adv_entries = []
    for attack in ('fgsm', 'pgd'):
        adv_dir = os.path.join(adv_root, attack, 'adv_real', id_name)
        if not os.path.isdir(adv_dir):
            continue
        for pair in sorted(os.listdir(adv_dir)):
            adv_img = os.path.join(adv_dir, pair, 'adv.png')
            sim_txt = os.path.join(adv_dir, pair, 'sim.txt')
            if not os.path.isfile(adv_img):
                continue
            sim = int(open(sim_txt).read().strip()) if os.path.isfile(sim_txt) else 50
            rel = os.path.relpath(adv_img).replace('\\', '/')
            if not rel.startswith('./'):
                rel = './' + rel
            prompt = (
                f"<|face_pad|>Please determine whether the person in the input image is VIP user. "
                f"The face similarity between the input face and VIP user is {sim}/100. "
                f"The face tokens are shown as follows, <|face_pad|> . "
                f"You should first give your answer by 'yes' or 'no'. "
                f"Then, you should explain your reasoning step by step based on different facial attributes."
            )
            entry = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<Conclusion>[Yes] The two images are of the same person.</Conclusion>\n"}
                ],
                "images": [rel]
            }
            adv_entries.append(entry)

    n_adv = max(1, int(len(original) * adv_fraction))
    rng = random.Random(seed)
    selected_adv = rng.sample(adv_entries, min(n_adv, len(adv_entries))) if adv_entries else []

    combined = original + selected_adv
    rng.shuffle(combined)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"  Built mixed JSON: {len(original)} original + {len(selected_adv)} adv -> {output_path}")
    return True


def resolve_training_json(id_name: str, adv_root: str, json_dir: str, tmp_dir: str,
                          adv_fraction: float, seed: int,
                          clean_only: bool = False) -> Optional[str]:
    """Return the path to the training JSON to use for this VIP."""
    # When --clean_only, skip adversarial data and go straight to original JSON
    if clean_only:
        original = os.path.join(json_dir, f'{id_name}.json')
        if os.path.isfile(original):
            print(f"  [CLEAN ONLY] Using original Stage3 JSON: {original}")
            return original
        return None

    # Priority 1: pre-built mixed JSON from generate_adversarial_dataset.py
    pre_built = os.path.join(adv_root, 'json', f'{id_name}_adv_mixed.json')
    if os.path.isfile(pre_built):
        return pre_built

    # Priority 2: build one now from original + adversarial components
    tmp_path = os.path.join(tmp_dir, f'{id_name}_adv_mixed.json')
    ok = build_mixed_json_if_missing(
        id_name=id_name,
        adv_root=adv_root,
        json_dir=json_dir,
        output_path=tmp_path,
        adv_fraction=adv_fraction,
        seed=seed,
    )
    if ok and os.path.isfile(tmp_path):
        return tmp_path

    # Priority 3: fall back to original Stage3 JSON
    original = os.path.join(json_dir, f'{id_name}.json')
    if os.path.isfile(original):
        print(f"  [WARN] No adversarial data found. Falling back to original JSON.")
        return original

    return None


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model_for_training(
    pretrain_path: str,
    vip_token_path: str,
    token_num: int,
    optim_facechecker: bool = False,
    scratch: bool = False,
) -> tuple:
    """
    Load VIPGuard with pretrained VIP tokens ready for adversarial fine-tuning.

    Frozen:   Qwen2.5-VL LLM, VisionTransformer, TransFace
    Trainable: vip_prompt only (or also FaceChecker if optim_facechecker=True)

    Uses 4-bit NF4 quantization (bitsandbytes) to fit the model entirely on GPU
    and avoid disk offloading which makes training impractically slow.
    """
    from transformers import BitsAndBytesConfig
    from Models.VIPGuard import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from Models.Wrapper import Wrapper

    print(f"  Loading VLM from {pretrain_path} ...")

    # 4-bit NF4 quantization: reduces 14GB model to ~7.4GB so it fits on the GPU.
    # bnb_4bit_compute_dtype=bfloat16 keeps all model layers in BF16 (dtype-consistent).
    # Numerical stability is handled by: AdamW eps=1e-2, grad_clip=1.0, and NaN guard.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs = dict(
        device_map={'': 'cuda:0'},
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrain_path, **load_kwargs
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(pretrain_path)

    model = Wrapper(
        vl_model=vl_model,
        processor=processor,
        optim_facechecker=optim_facechecker,
        vip_token_num=token_num,
    )
    # FaceChecker is not quantized — cast to bfloat16 to match model compute dtype
    model.vl_model.facechecker.to(torch.bfloat16)

    if scratch:
        # From-scratch: keep the random nn.init.normal_ initialization from Wrapper.
        # Do NOT load pretrained tokens — tokens start fresh and jointly learn
        # clean identity + adversarial robustness from step 1.
        print(f"  [SCRATCH] Starting from random VIP token initialization.")
    else:
        load_vip_token_into(model, vip_token_path)

    # Gradient checkpointing: frees intermediate activations during forward and
    # recomputes them during backward. Critical for 16GB VRAM — without it,
    # all 28 transformer layer activations stay in VRAM simultaneously.
    model.vl_model.gradient_checkpointing_enable()
    # Disable KV cache: not needed for training (generation only), wastes VRAM.
    model.vl_model.config.use_cache = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_trainable:,}")
    return model, processor


# ─── Training ─────────────────────────────────────────────────────────────────

def adversarial_train_one_id(
    id_name: str,
    json_path: str,
    pretrain_path: str,
    vip_token_path: str,
    token_num: int,
    epochs: int,
    lr: float,
    gradient_accumulation_step: int,
    save_path: str,
    use_mixed_precision: bool,
    mixed_precision_dtype: str,
    optim_facechecker: bool,
    scratch: bool = False,
):
    """
    Train VIP tokens for one identity on the mixed adversarial dataset.

    When scratch=True, tokens start from random initialization instead of
    loading pretrained weights — allowing joint clean+adversarial learning
    without any inherited fine-tuning bias.

    Imports and calls Stage3_train.train_one_epoch directly to reuse all
    existing training logic (mixed precision, gradient accumulation, checkpointing).
    """
    from Stage3_train import train_one_epoch
    from VIP_Dataset import VIP_Dataset

    model, processor = load_model_for_training(
        pretrain_path=pretrain_path,
        vip_token_path=vip_token_path,
        token_num=token_num,
        optim_facechecker=optim_facechecker,
        scratch=scratch,
    )

    print(f"\n  Loading dataset from: {json_path}")
    train_set = VIP_Dataset(json_path, processor=processor)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        num_workers=0,   # workers spawn extra processes that consume RAM on 16GB systems
        pin_memory=False,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.get_optim_params(),
        lr=lr,
        weight_decay=1e-3,
        eps=1e-2,  # large eps prevents denominator blowup when gradient is near-zero
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_loader) // gradient_accumulation_step),
        eta_min=lr * 0.0001,
    )

    # Build args namespace matching train_one_epoch's requirements exactly
    args = argparse.Namespace(
        name=id_name,
        use_mixed_precision=use_mixed_precision,
        mixed_precision_dtype=mixed_precision_dtype,
        gradient_accumulation_step=gradient_accumulation_step,
    )

    print(f"  Starting adversarial fine-tuning: {epochs} epoch(s), lr={lr}")
    for epoch in range(epochs):
        print(f"\n  --- Epoch {epoch+1}/{epochs} ---")
        train_one_epoch(
            model=model,
            processor=processor,
            train_loader=train_loader,
            optimizer=optimizer,
            args=args,
            epoch=epoch + 1,
            scheduler=scheduler,
            save_base_path=save_path,
        )
        torch.cuda.empty_cache()  # clear bitsandbytes dequantized weight caches between epochs

    print(f"  Adversarial fine-tuning complete. Checkpoint: {save_path}/vip_token.pt")

    # Free VRAM
    del model
    torch.cuda.empty_cache()


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarially fine-tune VIPGuard VIP tokens for robustness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ids', type=str, default='all',
                        help='Comma-separated VIP IDs or "all"')
    parser.add_argument('--pretrain_path', type=str,
                        default='./checkpoints/checkpoints_attr_Stage2_merge',
                        help='Path to Stage2 pretrained checkpoint')
    parser.add_argument('--vip_token_dir', type=str,
                        default='./FaceDATA/Pretrained_VIPToken',
                        help='Directory with per-VIP pretrained vip_token .pt files')
    parser.add_argument('--adv_root', type=str, default='./data/adversarial',
                        help='Root directory of adversarial images and JSONs')
    parser.add_argument('--json_dir', type=str, default='./FaceDATA/Stage3_training_json',
                        help='Directory of original Stage3 training JSONs')
    parser.add_argument('--token_num', type=int, default=32,
                        help='Number of VIP tokens')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of adversarial fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (lower than original 1.0 to preserve clean performance)')
    parser.add_argument('--gradient_accumulation_step', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device index or "cpu"')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--mixed_precision_dtype', type=str, default='bf16',
                        choices=['fp16', 'bf16'])
    parser.add_argument('--adv_fraction', type=float, default=0.25,
                        help='Fraction of adversarial samples in mixed JSON')
    parser.add_argument('--optim_facechecker', action='store_true',
                        help='Also fine-tune FaceChecker weights (more aggressive)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/Stage3_adv',
                        help='Root directory for saving fine-tuned checkpoints')
    parser.add_argument('--scratch', action='store_true',
                        help='Train from random VIP token init instead of loading pretrained tokens')
    parser.add_argument('--clean_only', action='store_true',
                        help='Skip adversarial data entirely — train on original Stage3 JSON only')
    return parser.parse_args()



# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.ids.lower() == 'all':
        ids_to_train = ALL_IDS
    else:
        ids_to_train = [x.strip() for x in args.ids.split(',')]

    print(f"Adversarial training for {len(ids_to_train)} VIP identities")
    print(f"LR: {args.lr}  |  Epochs: {args.epochs}  |  Adv fraction: {args.adv_fraction}")

    tmp_dir = os.path.join(args.adv_root, 'json')
    os.makedirs(tmp_dir, exist_ok=True)

    for id_name in ids_to_train:
        print(f"\n{'='*60}")
        print(f"Adversarial training: {id_name}")
        print(f"{'='*60}")

        # Resolve VIP token path (.pt file or checkpoint directory)
        vip_token_path = os.path.join(args.vip_token_dir, f'{id_name}.pt')
        if not os.path.isfile(vip_token_path):
            vip_token_path = os.path.join(args.vip_token_dir, id_name, 'vip_token.pt')
        if not os.path.isfile(vip_token_path):
            print(f"  [SKIP] VIP token not found for {id_name}")
            continue

        # Resolve training JSON
        json_path = resolve_training_json(
            id_name=id_name,
            adv_root=args.adv_root,
            json_dir=args.json_dir,
            tmp_dir=tmp_dir,
            adv_fraction=args.adv_fraction,
            seed=42,
            clean_only=getattr(args, 'clean_only', False),
        )
        if json_path is None:
            print(f"  [SKIP] No training JSON available for {id_name}")
            continue

        save_path = os.path.join(args.save_dir, id_name)

        adversarial_train_one_id(
            id_name=id_name,
            json_path=json_path,
            pretrain_path=args.pretrain_path,
            vip_token_path=vip_token_path,
            token_num=args.token_num,
            epochs=args.epochs,
            lr=args.lr,
            gradient_accumulation_step=args.gradient_accumulation_step,
            save_path=save_path,
            use_mixed_precision=args.use_mixed_precision,
            mixed_precision_dtype=args.mixed_precision_dtype,
            optim_facechecker=args.optim_facechecker,
            scratch=args.scratch,
        )

    print(f"\nAll adversarial fine-tuning complete. Checkpoints in: {args.save_dir}")


if __name__ == '__main__':
    main()
