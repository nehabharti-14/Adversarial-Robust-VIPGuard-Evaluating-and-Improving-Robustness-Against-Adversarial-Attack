"""
VIPGuard from-scratch adversarial training monitor.
Writes a detailed status block to pipeline_monitor.log every 60 seconds.
Only one instance can run at a time (enforced by PID file).

Tracks:
  - Per-ID training progress (epoch, step, loss)
  - Clean loss vs adversarial loss breakdown
  - VIP token norm change (confirms tokens are actually moving)
  - Checkpoint save times and shapes
  - Live evaluation progress when running
  - Final results summary

Usage: python monitor_pipeline.py
"""

import json
import os
import re
import sys
import tempfile
import time

_TMP      = tempfile.gettempdir()
PIDFILE   = os.path.join(_TMP, "vipguard_monitor.pid")
LOG       = "e:/VIPGuard/pipeline_monitor.log"
CKPT_DIR  = "e:/VIPGuard/checkpoints/Stage3_scratch"
EVAL_JSON = "e:/VIPGuard/results/after_scratch_train/robustness_results.json"
EVAL_LOG  = os.path.join(_TMP, "eval_scratch.log")
INTERVAL  = 60

ID_LOGS = {
    "id0": os.path.join(_TMP, "scratch_train_id0.log"),
    "id1": os.path.join(_TMP, "scratch_train_id1.log"),
    "id2": os.path.join(_TMP, "scratch_train_id2.log"),
}

CRASH_PAT = re.compile(r"RuntimeError|CUDA error|out of memory|CUBLAS_STATUS")
EPOCH_PAT = re.compile(r"Epoch (\d+):")
STEP_PAT  = re.compile(r"\|\s+(\d+/\d+)")
LOSS_PAT  = re.compile(r"Loss VQA:\s+([\d.]+)")
ADV_LOSS_PAT  = re.compile(r"Loss VQA:\s+([\d.]+).*adv_real")
CLEAN_LOSS_PAT = re.compile(r"Loss VQA:\s+([\d.]+).*(?:r_r_i|r_r_d_i|r_fs_i|r_efs_i)")
NORM_PAT  = re.compile(r"Prompt norm change:\s+([\d.]+)")


def check_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def enforce_single_instance():
    if os.path.exists(PIDFILE):
        try:
            existing = int(open(PIDFILE).read().strip())
            if check_pid(existing):
                print(f"Monitor already running (PID {existing}). Exiting.")
                sys.exit(1)
        except (ValueError, IOError):
            pass
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))


def parse_log(path: str) -> dict:
    result = {
        "epoch": None, "step": None,
        "last_loss": None, "errors": 0, "state": "not_started",
        "adv_losses": [], "clean_losses": [], "norm_changes": [],
    }
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return result
    try:
        text = open(path, errors="replace").read()
    except IOError:
        return result

    if CRASH_PAT.search(text):
        result["errors"] = len(CRASH_PAT.findall(text))

    epoch_matches = EPOCH_PAT.findall(text)
    if not epoch_matches:
        result["state"] = "loading"
        return result

    result["state"] = "running"
    result["epoch"] = epoch_matches[-1]

    step_matches = STEP_PAT.findall(text)
    if step_matches:
        result["step"] = step_matches[-1]

    loss_matches = LOSS_PAT.findall(text)
    if loss_matches:
        result["last_loss"] = loss_matches[-1]

    # Separate adversarial vs clean losses for last 50 lines
    lines = text.splitlines()[-50:]
    for line in lines:
        m = re.search(r"Loss VQA:\s+([\d.]+).*adv_real", line)
        if m:
            result["adv_losses"].append(float(m.group(1)))
        m2 = re.search(r"Loss VQA:\s+([\d.]+).*(?:r_r_i|r_r_d_i|r_fs_i|r_efs_i)", line)
        if m2:
            result["clean_losses"].append(float(m2.group(1)))

    norm_matches = NORM_PAT.findall(text)
    if norm_matches:
        result["norm_changes"] = [float(x) for x in norm_matches[-5:]]

    if "Adversarial fine-tuning complete" in text or "Training complete!" in text:
        result["state"] = "complete"

    return result


def ckpt_info(id_name: str) -> str:
    path = os.path.join(CKPT_DIR, id_name, "vip_token.pt")
    if os.path.exists(path):
        mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(path)))
        try:
            import torch
            state = torch.load(path, map_location="cpu")
            token = state.get("vip_token", next(iter(state.values()))) if isinstance(state, dict) else state
            shape_str = f"{list(token.shape)}"
        except Exception:
            shape_str = "?"
        return f"[DONE]  {id_name}  shape={shape_str}  (saved {mtime})"
    return f"[WAIT]  {id_name}"


def format_id_status(id_name: str) -> list:
    info = parse_log(ID_LOGS[id_name])
    state = info["state"]
    lines = []

    err_str = f"  *** CRASH ({info['errors']} errors) ***" if info["errors"] else ""

    if state == "complete":
        lines.append(f"  {id_name}: COMPLETE ✓{err_str}")
    elif state == "running":
        lines.append(f"  {id_name}: epoch {info['epoch']}/5  step {info['step']}  loss={info['last_loss']}{err_str}")
        # Show clean vs adversarial loss breakdown
        if info["adv_losses"]:
            avg_adv = sum(info["adv_losses"]) / len(info["adv_losses"])
            lines.append(f"    adv_loss  (last {len(info['adv_losses'])} samples): avg={avg_adv:.4f}")
        if info["clean_losses"]:
            avg_clean = sum(info["clean_losses"]) / len(info["clean_losses"])
            lines.append(f"    clean_loss (last {len(info['clean_losses'])} samples): avg={avg_clean:.4f}")
        # Show token norm changes (confirms learning is happening)
        if info["norm_changes"]:
            avg_norm = sum(info["norm_changes"]) / len(info["norm_changes"])
            lines.append(f"    token_norm_change (last {len(info['norm_changes'])} steps): avg={avg_norm:.4f}  [learning signal]")
    elif state == "loading":
        lines.append(f"  {id_name}: loading model...")
    else:
        lines.append(f"  {id_name}: not started yet")

    return lines


def eval_is_running() -> bool:
    if not os.path.exists(EVAL_LOG):
        return False
    if not os.path.exists(EVAL_JSON):
        return os.path.getsize(EVAL_LOG) > 0
    return os.path.getmtime(EVAL_LOG) > os.path.getmtime(EVAL_JSON)


def eval_progress() -> list:
    if not os.path.exists(EVAL_LOG) or os.path.getsize(EVAL_LOG) == 0:
        return []
    try:
        lines = open(EVAL_LOG, errors="replace").readlines()
    except IOError:
        return []
    useful = [l.rstrip() for l in lines if l.strip() and
              any(k in l for k in ("pred=", "Category:", "Evaluating:", "===", "DONE", "FAILED"))]
    return useful[-10:] if useful else []


def eval_results() -> list:
    if not os.path.exists(EVAL_JSON):
        return ["  Evaluation: waiting for all training to complete..."]
    try:
        r = json.load(open(EVAL_JSON))
        lines = ["  Results:"]
        for id_name, cats in sorted(r.items()):
            lines.append(f"    {id_name}:")
            for cat, v in cats.items():
                acc = v.get("accuracy")
                if acc is not None:
                    bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                    lines.append(f"      {cat:<14} {acc*100:5.1f}%  {bar}")
        return lines
    except Exception:
        return ["  Results: (error reading JSON)"]


def write_status():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "━" * 60,
        f"[{ts}] CURRICULUM ADVERSARIAL TRAINING",
        "  Stage1: LR=1.0 | epochs=3 | clean data | scratch init",
        "  Stage2: LR=0.1 | epochs=3 | adv 15%   | from Stage1 ckpt",
        "",
    ]

    # Training progress
    lines.append("  Training (id0 → id1 → id2):")
    for id_name in ("id0", "id1", "id2"):
        lines.extend(format_id_status(id_name))

    # Checkpoints
    lines += ["", "  Checkpoints:"]
    for id_name in ("id0", "id1", "id2"):
        lines.append(f"    {ckpt_info(id_name)}")

    # Evaluation
    lines.append("")
    if eval_is_running():
        lines.append("  Evaluation (live):")
        for l in eval_progress():
            lines.append(f"    {l}")
    else:
        lines.extend(eval_results())

    lines.append("")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    enforce_single_instance()
    try:
        print(f"Monitor started (PID {os.getpid()}). Writing to {LOG} every {INTERVAL}s.")
        write_status()
        while True:
            time.sleep(INTERVAL)
            write_status()
    finally:
        try:
            os.remove(PIDFILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
