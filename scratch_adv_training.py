"""
Two-stage curriculum adversarial training for VIPGuard.

WHY TWO STAGES:
  FGSM attack sim range (47–75) and deepfake sim range (52–78) overlap almost completely.
  Any single-stage training with adversarial examples caused the model to either:
    - Ignore low sim scores entirely (orig_sim strategy) → 0% FGSM/PGD at eval
    - Say Yes to everything in the medium sim range (adv_sim strategy) → 0% deepfake

  The root cause: the 32B LLM is frozen and uses the text sim score as the primary signal.
  VIP tokens must override this via visual features from Qwen2.5-VL's vision encoder.
  FGSM/PGD attacks target TransFace, not Qwen2.5-VL — so the VLM still sees the real
  face clearly even in adversarially perturbed images.

STAGE 1 — Clean training (LR=1.0, epochs=3, no adversarial data):
  VIP tokens learn from scratch to:
    - Authenticate real VIP images visually (normal_real ~100%)
    - Reject deepfakes using visual features, not just sim score (~90-100%)
  This builds a solid foundation of visual discrimination.

STAGE 2 — Adversarial fine-tuning (LR=0.1, epochs=3, adv_fraction=0.15):
  Continues from Stage 1 checkpoint. Teaches:
    "Even when the sim text says 47–75/100, trust your visual assessment of the VIP tokens"
  Low LR (10x smaller) ensures Stage 1 visual discrimination is preserved.
  Small adv_fraction (15%) keeps adversarial signal light — just enough to add robustness.

Output: checkpoints/Stage3_scratch/id{k}/vip_token.pt
Eval:   results/after_scratch_train/robustness_results.json
"""

import json
import os
import subprocess
import sys
import tempfile
import time

PYTHON   = sys.executable
BASE     = "e:/VIPGuard"
LOG      = os.path.join(BASE, "pipeline_monitor.log")
CKPT_DIR = os.path.join(BASE, "checkpoints", "Stage3_scratch")
_TMP     = tempfile.gettempdir()

# ─── Shared args ─────────────────────────────────────────────────────────────
COMMON_ARGS = [
    "--pretrain_path", os.path.join(BASE, "checkpoints", "checkpoints_attr_Stage2_merge"),
    "--vip_token_dir", os.path.join(BASE, "FaceDATA", "Pretrained_VIPToken"),
    "--adv_root",      os.path.join(BASE, "data", "adversarial"),
    "--json_dir",      os.path.join(BASE, "FaceDATA", "Stage3_training_json"),
    "--token_num",     "32",
    "--gradient_accumulation_step", "8",
    "--device",        "0",
    "--use_mixed_precision",
    "--mixed_precision_dtype", "bf16",
    "--save_dir",      CKPT_DIR,
]

# Stage 1: clean data only, high LR, from scratch
STAGE1_ARGS = COMMON_ARGS + [
    "--epochs",        "3",
    "--lr",            "1.0",
    "--adv_fraction",  "0.0",
    "--scratch",       # random VIP token init
    "--clean_only",    # skip adversarial JSON, use original Stage3 JSON
]

# Stage 2: mixed adversarial data, low LR, continues from Stage 1 checkpoint
STAGE2_ARGS = COMMON_ARGS + [
    "--epochs",        "3",
    "--lr",            "0.1",   # 10x lower — preserves Stage 1 visual discrimination
    "--adv_fraction",  "0.15",  # light adversarial pressure (13% of data)
    # no --scratch: loads Stage 1 checkpoint from save_dir
    # no --clean_only: uses pre-built mixed JSON with adv_sim prompts
]


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ckpt_exists(id_name: str) -> bool:
    return os.path.exists(os.path.join(CKPT_DIR, id_name, "vip_token.pt"))


def run_stage(stage: int, id_name: str, extra_args: list) -> bool:
    """Run one training stage for one ID. Returns True on success."""
    log_path = os.path.join(_TMP, f"scratch_train_{id_name}.log")
    stage_label = f"Stage{stage}"
    lr   = extra_args[extra_args.index("--lr") + 1]   if "--lr"     in extra_args else "?"
    ep   = extra_args[extra_args.index("--epochs") + 1] if "--epochs" in extra_args else "?"
    data = "clean" if "--clean_only" in extra_args else "mixed(adv)"

    log(f">>> {id_name} {stage_label}: {data}  LR={lr}  epochs={ep}")

    cmd = [PYTHON, "-u", os.path.join(BASE, "adversarial_training.py"),
           "--ids", id_name] + extra_args

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Stage 2 should load from Stage 1 checkpoint via --vip_token_dir pointing at CKPT_DIR
    # Override vip_token_dir to point at our scratch checkpoint for Stage 2
    if stage == 2:
        idx = cmd.index("--vip_token_dir")
        cmd[idx + 1] = CKPT_DIR

    open_mode = "w" if stage == 1 else "a"
    with open(log_path, open_mode) as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=BASE)

    if proc.returncode == 0 and ckpt_exists(id_name):
        log(f">>> {id_name} {stage_label}: DONE — checkpoint saved.")
        return True
    else:
        log(f">>> {id_name} {stage_label}: FAILED (exit {proc.returncode})")
        try:
            tail = open(log_path, errors="replace").readlines()
            for line in tail[-20:]:
                log(f"  {line.rstrip()}")
        except IOError:
            pass
        return False


def train_id(id_name: str) -> bool:
    """Run both stages for one VIP identity."""
    log(f"{'='*60}")
    log(f"=== {id_name}: CURRICULUM TRAINING ===")

    # Stage 1: clean visual foundation
    ok1 = run_stage(1, id_name, STAGE1_ARGS)
    if not ok1:
        log(f"WARNING: {id_name} Stage1 failed — skipping Stage2.")
        return False

    time.sleep(5)  # brief pause between stages

    # Stage 2: gentle adversarial fine-tuning on top of Stage 1
    ok2 = run_stage(2, id_name, STAGE2_ARGS)
    return ok2


def run_evaluation():
    log(">>> Evaluation: id0, id1, id2 — curriculum-trained VIP tokens")
    eval_log = os.path.join(_TMP, "eval_scratch.log")
    cmd = [
        PYTHON, "-u", os.path.join(BASE, "evaluate_robustness.py"),
        "--ids",           "id0,id1,id2",
        "--pretrain_path", os.path.join(BASE, "checkpoints", "checkpoints_attr_Stage2_merge"),
        "--vip_token_dir", CKPT_DIR,
        "--adv_root",      os.path.join(BASE, "data", "adversarial"),
        "--token_num",     "32",
        "--device",        "0",
        "--output_dir",    os.path.join(BASE, "results", "after_scratch_train"),
        "--save_plots",
        "--max_per_category", "30",
    ]
    with open(eval_log, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=BASE)

    if proc.returncode != 0:
        log(f">>> Evaluation FAILED (exit {proc.returncode})")
        try:
            tail = open(eval_log, errors="replace").readlines()
            for line in tail[-20:]:
                log(f"  {line.rstrip()}")
        except IOError:
            pass
        return

    log(">>> Evaluation DONE")
    results_path = os.path.join(BASE, "results", "after_scratch_train", "robustness_results.json")
    if os.path.exists(results_path):
        r = json.load(open(results_path))
        log("=" * 60)
        log("CURRICULUM TRAINING RESULTS — Stage1(LR=1.0,ep=3,clean) + Stage2(LR=0.1,ep=3,adv=15%)")
        log("=" * 60)
        for id_name, cats in sorted(r.items()):
            log(f"  {id_name}:")
            for cat, v in cats.items():
                acc = v.get("accuracy")
                if acc is not None:
                    log(f"    {cat}: {acc*100:.1f}%")


def main():
    log("=" * 60)
    log("=== CURRICULUM ADVERSARIAL TRAINING ===")
    log("    Stage1: clean, LR=1.0, epochs=3, scratch init")
    log("    Stage2: mixed(adv 15%), LR=0.1, epochs=3, from Stage1 ckpt")
    log("    adv_sim in prompts (matches eval distribution)")
    log("=" * 60)

    for id_name in ("id0", "id1", "id2"):
        ok = train_id(id_name)
        if not ok:
            log(f"WARNING: {id_name} failed — continuing with next ID.")
        time.sleep(10)

    run_evaluation()
    open(os.path.join(_TMP, "scratch_complete"), "w").close()
    log("=== CURRICULUM TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()
