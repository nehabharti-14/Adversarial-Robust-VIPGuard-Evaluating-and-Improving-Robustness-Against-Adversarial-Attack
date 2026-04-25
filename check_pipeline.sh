#!/bin/bash
# VIPGuard pipeline monitor — single instance, appends to pipeline_monitor.log every 5 min.
# Usage: bash check_pipeline.sh

PIDFILE="/c/Users/TL1/AppData/Local/Temp/vipguard_monitor.pid"
LOG="e:/VIPGuard/pipeline_monitor.log"
PYTHON="/c/Users/TL1/anaconda3/envs/myexamenv/python.exe"
INTERVAL=300

# Enforce single instance via PID file
if [ -f "$PIDFILE" ]; then
    existing=$(cat "$PIDFILE")
    if ps -p "$existing" > /dev/null 2>&1; then
        echo "Monitor already running (PID $existing). Exiting."
        exit 1
    fi
fi
echo $$ > "$PIDFILE"
trap "rm -f $PIDFILE" EXIT

write_status() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    {
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[$ts] PIPELINE STATUS"
        echo ""

        # Training process
        local pid
        pid=$(ps aux 2>/dev/null | grep "myexamenv/python" | grep -v grep | awk '{print $1}' | head -1)
        if [ -n "$pid" ]; then
            echo "  Training process: RUNNING (PID $pid)"
        else
            echo "  Training process: not running"
        fi

        echo ""

        # Per-identity progress
        for id in id1 id2; do
            local logfile="/tmp/adv_train_${id}_v8.log"
            local epoch_lines=0
            [ -f "$logfile" ] && [ -s "$logfile" ] && epoch_lines=$(grep -cE "Epoch [12]:" "$logfile" 2>/dev/null || echo 0)

            if [ "$epoch_lines" -gt 0 ] 2>/dev/null; then
                local epoch step loss errors
                epoch=$(grep -oE "Epoch [12]:" "$logfile" | grep -oE "[12]" | tail -1)
                step=$(grep -oE "\| +[0-9]+/[0-9]+" "$logfile" | grep -oE "[0-9]+/[0-9]+" | tail -1)
                loss=$(grep -oE "Loss VQA: [0-9]+\.[0-9]+" "$logfile" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
                errors=$(grep -cE "RuntimeError|CUDA error|out of memory|CUBLAS_STATUS" "$logfile" 2>/dev/null || echo 0)
                if [ "$errors" -gt 0 ]; then
                    echo "  $id: epoch ${epoch:-?}/2  step ${step:-?}  loss=${loss:-?}  *** CRASH ($errors errors) ***"
                else
                    echo "  $id: epoch ${epoch:-?}/2  step ${step:-?}  loss=${loss:-?}"
                fi
            elif [ -f "$logfile" ] && [ -s "$logfile" ]; then
                echo "  $id: loading model..."
            else
                echo "  $id: not started yet"
            fi
        done

        echo ""

        # Checkpoints
        echo "  Checkpoints:"
        for id in id0 id1 id2; do
            local ckpt="e:/VIPGuard/checkpoints/Stage3_adv/$id/vip_token.pt"
            if [ -f "$ckpt" ]; then
                local saved
                saved=$(stat -c '%y' "$ckpt" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
                echo "    [DONE] $id  (saved $saved)"
            else
                echo "    [WAIT] $id"
            fi
        done

        echo ""

        # Post-training evaluation results
        local eval_json="e:/VIPGuard/results/after_adv_train/robustness_results.json"
        if [ -f "$eval_json" ]; then
            echo "  Post-training eval:"
            $PYTHON - << 'PYEOF' 2>/dev/null
import json
r = json.load(open("e:/VIPGuard/results/after_adv_train/robustness_results.json"))
for k, cats in sorted(r.items()):
    print(f"    {k}:")
    for cat, v in cats.items():
        acc = v.get("accuracy")
        if acc is not None:
            print(f"      {cat}: {acc*100:.1f}%")
PYEOF
        else
            echo "  Post-training eval: waiting"
        fi

        echo ""
    } >> "$LOG"
}

echo "Monitor started (PID $$). Writing to $LOG every 5 min."
write_status
while true; do
    sleep "$INTERVAL"
    write_status
done
