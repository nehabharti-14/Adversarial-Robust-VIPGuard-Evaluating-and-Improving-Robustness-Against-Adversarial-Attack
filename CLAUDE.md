# CLAUDE.md — VIPGuard Project Documentation

This file is persistent documentation for Claude Code and future developers. It covers
the full project architecture, the adversarial robustness extension, and step-by-step
instructions for running everything.

---

## Table of Contents

1. [What VIPGuard Does](#1-what-vipguard-does)
2. [System Architecture](#2-system-architecture)
3. [Data Layout](#3-data-layout)
4. [Training Stages](#4-training-stages)
5. [Adversarial Robustness Extension](#5-adversarial-robustness-extension)
6. [File Structure Overview](#6-file-structure-overview)
7. [How to Run Everything](#7-how-to-run-everything)
8. [Key Implementation Pitfalls](#8-key-implementation-pitfalls)
9. [Evaluation Metrics Reference](#9-evaluation-metrics-reference)

---

## 1. What VIPGuard Does

VIPGuard is a **VIP face verification and deepfake detection** system. Instead of a
simple binary classifier, it uses a Vision-Language Model (VLM) to reason about facial
attributes and identity.

**Core question it answers:** "Is the person in this image the registered VIP, or is it
a deepfake / impostor?"

**Why a VLM?** Traditional face verification models produce a similarity score but no
reasoning. VIPGuard adds VIP-specific learned tokens to a pretrained VLM (Qwen2.5-VL)
so the model can explain its decision in natural language.

**Output format:**
```
<Conclusion>[Yes] The two images are of the same person. The jawline structure,
eye spacing, and skin tone match the registered VIP profile.</Conclusion>
```

---

## 2. System Architecture

### Inference Pipeline (step by step)

```
Input Image (any size)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Face Similarity Scoring (TransFace)                │
│                                                             │
│  img → cv2.resize(112,112) → normalize([-1,1])             │
│      → TransFace ViT → 512-dim embedding                   │
│      → cosine_similarity(emb, VIP_center)                  │
│      → similarity_score: int [0, 100]                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: VIPGuard VLM Inference                             │
│                                                             │
│  Prompt: "<|face_pad|> <image>                             │
│           Similarity {score}/100. Is this VIP person?"     │
│                                                             │
│  Image → Qwen2.5-VL Visual Encoder → image_features        │
│  <|face_pad|> → FaceChecker(VIP_tokens, image_features)    │
│               → face_embed_tokens                          │
│  [face_embed_tokens + image_tokens + text_tokens]          │
│  → Qwen2.5-VL LLM (80 layers, frozen)                     │
│  → Generate: "<Conclusion>[Yes/No] ..."                    │
└─────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Location | Role |
|-----------|----------|------|
| **TransFace** | `Models/Face_Model/TransFace/` | Face embedding (512-dim), used for similarity scoring |
| **FaceChecker** | `Models/VIPGuard/FaceChecker.py` | CrossAttention between VIP tokens and face image features |
| **Qwen2.5-VL VLM** | `checkpoints/checkpoints_attr_Stage2_merge/` | 32B frozen language+vision model |
| **VIP Tokens** | `FaceDATA/Pretrained_VIPToken/id{k}.pt` | Learnable identity embeddings [N, 3584], one set per VIP |
| **Wrapper** | `Models/Wrapper.py` | Manages VIP token init/load/save, exposes `get_optim_params()` |

### FaceChecker Detail

```python
# Simplified forward pass
face_embeds = facechecker(
    img_feature_q=image_features,   # query: from vision encoder
    img_feature_v=vip_prompt        # value: learnable VIP tokens [N, 3584]
)
# face_embeds replace <|face_pad|> token positions in the LLM input
```

### VIP Token Checkpoint Format

```python
state = torch.load('vip_token.pt')
# state = {'vip_token': tensor([N, 3584])}
# Load: model.vl_model.facechecker.face_checker.vip_prompt.data = state['vip_token']
```

### Similarity Score Formula

```python
# Matches inference.ipynb exactly
emb  = F.normalize(transface_embedding, dim=0)    # (512,)
ctr  = F.normalize(vip_center, dim=0)             # (512,)
sim  = torch.dot(emb, ctr).item()                 # [-1, 1]
score = int((0.5 + 0.5 * sim) * 100)              # [0, 100]
```

---

## 3. Data Layout

```
VIPGuard/
├── FaceDATA/
│   ├── Training_Img/
│   │   └── id{0-21}/               # 22 VIP identities
│   │       ├── r_r_i/              # Real vs Real, SAME identity  → label=0 (authentic)
│   │       │   └── id{k}_{n}/
│   │       │       ├── 2.png       # Probe image (used for training/attacks)
│   │       │       └── real.png    # Reference image
│   │       ├── r_r_d_i/            # Real, DIFFERENT identity     → label=1
│   │       ├── r_fs_i/             # Face-Swapped deepfake         → label=1
│   │       │   └── id{k}_{n}/fake.png
│   │       └── r_efs_i/            # Entire-Face-Synthesis deepfake → label=1
│   │           └── id{k}_{n}/fake.png
│   ├── FaceEmb_Center/
│   │   └── id{k}.ckpt              # Raw tensor (512,) — VIP face centroid
│   ├── Pretrained_VIPToken/
│   │   └── id{k}.pt                # {'vip_token': tensor([N, 3584])}
│   └── Stage3_training_json/
│       └── id{k}.json              # Training data in Stage3 format
│
├── data/adversarial/               # Generated by generate_adversarial_dataset.py
│   ├── fgsm/adv_real/id{k}/{pair}/
│   │   ├── adv.png                 # Adversarial image (FGSM)
│   │   └── sim.txt                 # Similarity score of adversarial image
│   ├── pgd/adv_real/id{k}/{pair}/
│   │   ├── adv.png                 # Adversarial image (PGD)
│   │   └── sim.txt
│   └── json/
│       └── id{k}_adv_mixed.json    # Mixed training JSON (original + adversarial)
│
├── checkpoints/
│   ├── checkpoints_attr_Stage2_merge/  # Pretrained 32B Qwen2.5-VL (frozen)
│   ├── Stage3/                         # Standard fine-tuned VIP tokens
│   │   └── {name}/vip_token.pt
│   └── Stage3_adv/                     # Adversarially fine-tuned VIP tokens
│       └── {name}/vip_token.pt
│
└── results/                            # Evaluation outputs
    ├── before_adv_train/
    │   ├── robustness_results.json
    │   └── id{k}_sim_distribution.png
    └── after_adv_train/
        ├── robustness_results.json
        └── id{k}_sim_distribution.png
```

### Label Conventions (VIP_Dataset.py)

| Path substring | Label | Meaning |
|----------------|-------|---------|
| `r_r_i` | 0 | Authentic VIP (same-identity real pair) |
| `adv_real` | 0 | Authentic VIP (adversarially perturbed) |
| anything else | 1 | Impostor / deepfake |

---

## 4. Training Stages

### Stage 2: Pretrained Base Model
- **Script:** Uses ms-swift framework (external)
- **What it produces:** `checkpoints/checkpoints_attr_Stage2_merge/`
- **What it trains:** Qwen2.5-VL on general facial attribute understanding
- **Status:** Already done, checkpoint is present

### Stage 3: Per-VIP Token Fine-tuning
- **Script:** `Stage3_train.py`
- **What it trains:** Only VIP tokens (`vip_prompt`, shape [N, 3584]) — 32B LLM frozen
- **Loss:** Causal LM loss on answer tokens only (question tokens masked with -100)
- **Key params:** LR=1.0, AdamW, CosineAnnealingLR, epochs=1-5
- **Input:** `FaceDATA/Stage3_training_json/id{k}.json`
- **Output:** `checkpoints/Stage3/{name}/vip_token.pt`

```bash
python Stage3_train.py \
    --train_json_path ./FaceDATA/Stage3_training_json/id0.json \
    --name id0_train \
    --device 0 \
    --epoch 3 \
    --token_num 32 \
    --use_mixed_precision \
    --mixed_precision_dtype bf16
```

### Stage 3-ADV: Adversarial Fine-tuning (NEW)
- **Script:** `adversarial_training.py`
- **What it trains:** Same VIP tokens, but on mixed clean + adversarial data
- **Key difference:** LR=0.01 (10× lower), 25% adversarial samples
- **Input:** `data/adversarial/json/id{k}_adv_mixed.json`
- **Output:** `checkpoints/Stage3_adv/{name}/vip_token.pt`

---

## 5. Adversarial Robustness Extension

### Threat Model

- **Attacker goal:** Evade VIP detection — make a real VIP image look like a non-VIP
  (or make a deepfake pass as a real VIP)
- **Attack access:** White-box access to TransFace face embeddings
- **Perturbation constraint:** ε = 8/255 (pixel space), visually imperceptible

### Attack Types

#### FGSM (Fast Gradient Sign Method)
- **Type:** Untargeted dodge attack, single step
- **Formula:** `x_adv = x + ε · sign(∇_x cosine_sim(f(x), center))`
- **Strength:** Weak but fast; good for generating large adversarial datasets
- **Epsilon:** 8/255 pixel-space → 0.0627 normalized-space

#### PGD (Projected Gradient Descent)
- **Type:** Untargeted dodge attack, multi-step
- **Formula:** Iterative FGSM with projection back into ε-ball
- **Strength:** Stronger than FGSM; standard benchmark for adversarial robustness
- **Params:** ε=8/255, α=2/255, steps=10, random_start=True

### Attack Target

Both attacks target **TransFace embeddings** (not the full 32B VLM) because:
1. TransFace provides the similarity score that biases the LLM decision
2. Gradient computation through a 32B model is prohibitively slow
3. Attacking the face embedding effectively reduces the similarity score below threshold

### Adversarial Training Strategy

- Mix 75% original training data + 25% adversarial images
- VIP tokens learn to authenticate real VIP images even when perturbed
- Lower LR (0.01 vs 1.0) prevents overwriting existing clean-data knowledge
- FaceChecker optionally trainable via `--optim_facechecker`

### Expected Results

| Condition | Accuracy (typical) | FRR |
|-----------|-------------------|-----|
| Normal real | ~95–98% | ~2–5% |
| Deepfake | ~90–95% | — |
| FGSM attack (before training) | ~60–75% | ~25–40% |
| PGD attack (before training) | ~55–70% | ~30–45% |
| FGSM attack (after adv training) | ~80–90% | ~10–20% |
| PGD attack (after adv training) | ~75–88% | ~12–25% |

---

## 6. File Structure Overview

```
VIPGuard/
├── attacks/                        # NEW: Adversarial attack modules
│   ├── __init__.py
│   ├── fgsm.py                     # FGSM attack (single-step)
│   └── pgd.py                      # PGD attack (multi-step)
│
├── Models/
│   ├── Wrapper.py                  # VIP token management
│   ├── Face_Model/
│   │   ├── FaceModel.py            # FG_Face wrapper (note: face_emb_forward is @no_grad)
│   │   └── TransFace/backbones/
│   │       ├── vit.py              # VisionTransformer (returns 3-tuple: feat, weight, entropy)
│   │       └── iresnet.py          # Alternative ArcFace backbone
│   └── VIPGuard/
│       ├── FaceChecker.py          # CrossAttention for VIP verification
│       ├── modeling_qwen2_5_vl.py  # Full Qwen2.5-VL with face token injection
│       ├── processing_qwen2_5_vl.py
│       └── configuration_qwen2_5_vl.py
│
├── Stage3_train.py                 # Stage 3 training (train_one_epoch imported by adv_training)
├── VIP_Dataset.py                  # Dataset loader (PATCHED: adv_real → label=0)
│
├── generate_adversarial_dataset.py # NEW: Generate adversarial images for all VIPs
├── evaluate_robustness.py          # NEW: Evaluate on normal/deepfake/FGSM/PGD categories
├── adversarial_training.py         # NEW: Adversarially fine-tune VIP tokens
│
├── Inference.ipynb                 # Reference inference notebook
├── CLAUDE.md                       # This file
│
└── tool/
    ├── Crop_Method/                # Face cropping utilities
    └── Align_Method/               # Face alignment utilities
```

---

## 7. How to Run Everything

### Prerequisites

```bash
# Ensure these checkpoints are in place:
# ./checkpoints/checkpoints_attr_Stage2_merge/  (Qwen2.5-VL Stage2)
# ./Models/Face_Model/checkpoints/transface/glint360k_model_TransFace_L.pt
# ./FaceDATA/Pretrained_VIPToken/id{0-21}.pt
# ./FaceDATA/FaceEmb_Center/id{0-21}.ckpt
# ./FaceDATA/Training_Img/id{0-21}/...

pip install torch torchvision opencv-python matplotlib termcolor qwen-vl-utils
```

### Step 1: Generate Adversarial Images

Attacks run on TransFace only — no VLM needed, fast.

```bash
python generate_adversarial_dataset.py \
    --ids all \
    --attack both \
    --device 0 \
    --epsilon 0.03137 \
    --alpha 0.00784 \
    --steps 10 \
    --output_root ./data/adversarial

# Single ID test:
python generate_adversarial_dataset.py --ids id0 --attack fgsm --device 0
```

**Outputs:**
- `data/adversarial/fgsm/adv_real/id{k}/{pair}/adv.png`
- `data/adversarial/pgd/adv_real/id{k}/{pair}/adv.png`
- `data/adversarial/json/id{k}_adv_mixed.json`

### Step 2: Evaluate Robustness (Baseline — Before Adversarial Training)

Requires full VLM loaded. GPU with ≥24GB VRAM recommended.

```bash
python evaluate_robustness.py \
    --ids all \
    --pretrain_path ./checkpoints/checkpoints_attr_Stage2_merge \
    --vip_token_dir ./FaceDATA/Pretrained_VIPToken \
    --adv_root ./data/adversarial \
    --token_num 32 \
    --device 0 \
    --output_dir ./results/before_adv_train \
    --save_plots

# Single ID:
python evaluate_robustness.py --ids id0 --token_num 32 --device 0 --save_plots
```

**Outputs:**
- Console table with Accuracy / FAR / FRR / Performance Drop per category
- `results/before_adv_train/robustness_results.json`
- `results/before_adv_train/id{k}_sim_distribution.png`

### Step 3: Adversarial Fine-Tuning

Fine-tunes VIP tokens on mixed clean + adversarial data.

```bash
python adversarial_training.py \
    --ids all \
    --pretrain_path ./checkpoints/checkpoints_attr_Stage2_merge \
    --vip_token_dir ./FaceDATA/Pretrained_VIPToken \
    --adv_root ./data/adversarial \
    --token_num 32 \
    --epochs 2 \
    --lr 0.01 \
    --gradient_accumulation_step 8 \
    --device 0 \
    --use_mixed_precision \
    --mixed_precision_dtype bf16 \
    --adv_fraction 0.25

# Single ID:
python adversarial_training.py --ids id0 --token_num 32 --epochs 2 --device 0
```

**Outputs:** `checkpoints/Stage3_adv/id{k}/vip_token.pt`

### Step 4: Evaluate After Adversarial Training

```bash
python evaluate_robustness.py \
    --ids all \
    --pretrain_path ./checkpoints/checkpoints_attr_Stage2_merge \
    --vip_token_dir ./checkpoints/Stage3_adv \
    --adv_root ./data/adversarial \
    --token_num 32 \
    --device 0 \
    --output_dir ./results/after_adv_train \
    --save_plots
```

Compare `results/before_adv_train/` vs `results/after_adv_train/` to measure improvement.

### Quick Test (Single Image Inference)

Use `Inference.ipynb` — set `CONFIG["VIP_PATH"]` to any `vip_token.pt` checkpoint.

---

## 8. Key Implementation Pitfalls

### 1. `@torch.no_grad()` on `face_emb_forward`
`FG_Face.face_emb_forward()` ([Models/Face_Model/FaceModel.py:45](Models/Face_Model/FaceModel.py)) is decorated with `@torch.no_grad()`. Calling it during attack generation produces **zero gradients**.

**Fix:** Call the raw backbone directly:
```python
# WRONG (no gradients flow):
emb = fg_face.face_emb_forward(x)

# CORRECT (gradients flow through):
feat, _, _ = fg_face.face_model(x)   # returns 3-tuple
```

### 2. PGD Loop Memory Leak
Without `.detach()` between PGD iterations, the computation graph accumulates:
```python
# WRONG — OOM after a few steps:
for _ in range(num_steps):
    x_adv.requires_grad_(True)
    ...

# CORRECT — detach first:
for _ in range(num_steps):
    x_adv = x_adv.detach().requires_grad_(True)
    ...
```

### 3. TransFace BatchNorm in Training Mode
TransFace uses `BatchNorm1d`. With batch size=1, training mode produces degenerate statistics. Random patch masking also activates in training mode.

**Fix:** Always call `face_model.eval()` before attack forward passes.

### 4. Autocast Gradient Underflow
The TransFace `Attention` module uses `torch.cuda.amp.autocast(True)` internally. This can cause FP16 gradient underflow making attack gradients zero.

**Fix:** Wrap attack forward in `torch.cuda.amp.autocast(False)`:
```python
with torch.cuda.amp.autocast(False):
    feat, _, _ = face_backbone(x.float())
```

### 5. Center Tensor Format
`torch.load('FaceEmb_Center/id0.ckpt')` returns a **raw tensor**, not a dict:
```python
center = torch.load(path, map_location='cpu')  # tensor, shape (512,)
```

### 6. VIP Token Checkpoint Format
`torch.load('vip_token.pt')` returns a **dict** with key `'vip_token'`:
```python
state = torch.load('vip_token.pt')
# state = {'vip_token': tensor([N, 3584])}
model.vl_model.facechecker.face_checker.vip_prompt.data = state['vip_token']
```

### 7. `adv_real` Label Patch in VIP_Dataset
`VIP_Dataset.py:78` uses path substring matching for labels. Adversarial real images must have `adv_real` in their path **OR** the dataset must be patched.

**Applied patch** (already in the code):
```python
if 'r_r_i' in img_dir.lower() or 'adv_real' in img_dir.lower():
    label = 0   # Authentic VIP
else:
    label = 1   # Impostor / deepfake
```

### 8. `train_one_epoch` args Namespace
When importing `train_one_epoch` from `Stage3_train`, the `args` object must have exactly:
- `args.name` (str)
- `args.use_mixed_precision` (bool)
- `args.mixed_precision_dtype` (str: 'fp16' or 'bf16')
- `args.gradient_accumulation_step` (int)

### 9. GPU Memory — VLM is 32B Parameters
Loading the full Qwen2.5-VL requires ~32GB VRAM (FP16). Use `device_map='auto'` or `device_map='balanced'` to shard across GPUs. TransFace can run on CPU for similarity scoring.

---

## 9. Evaluation Metrics Reference

| Metric | Formula | When relevant |
|--------|---------|---------------|
| **Accuracy** | (TP + TN) / N | All categories |
| **FAR** (False Acceptance Rate) | FP / (FP + TN) | Deepfake / impostor categories |
| **FRR** (False Rejection Rate) | FN / (FN + TP) | Authentic VIP categories |
| **Performance Drop %** | (acc_baseline − acc_adv) / acc_baseline × 100 | Adversarial vs. normal |
| **Sim Mean / Std** | numpy mean/std of scores [0–100] | All categories |

**Confusion matrix convention:**
- Positive = model predicts "Yes" (is VIP)
- TP = real VIP correctly authenticated
- TN = deepfake/impostor correctly rejected
- FP = deepfake/impostor incorrectly authenticated (dangerous!)
- FN = real VIP incorrectly rejected (annoying but safe)

---

*Generated and maintained by Claude Code. Last updated: 2026-04-11.*
