#!/bin/bash
# Train the student on a RunPod pod, from an unpacked bundle in /root/spisev (cloud/pack.py).
#
#   DEADLINE_MIN=75 bash /root/spisev/pod_run.sh unsloth/Qwen3-1.7B qwen3-1.7b-v2 3
#
# The pod deletes itself DEADLINE_MIN minutes after this starts, finished or not, so a lost
# connection cannot leave it billing: pull runs/NAME before then. Logs go to /root/runs/NAME.log.
set -euo pipefail
BASE=$1 NAME=$2 EPOCHS=${3:-3}
DEADLINE_MIN=${DEADLINE_MIN:-75}
OUT=/root/runs/$NAME
mkdir -p "$OUT"

if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  nohup bash -c "sleep $((DEADLINE_MIN * 60)); runpodctl remove pod $RUNPOD_POD_ID" >/root/deadline.log 2>&1 &
  echo "deadline: pod $RUNPOD_POD_ID removes itself in $DEADLINE_MIN min"
else
  echo "deadline: runpodctl or RUNPOD_POD_ID missing, no self-removal" >&2
fi

cd /root/spisev
SPISEV_QWEN3=1 python3 setup_vm.py
python3 train_unsloth.py --base "$BASE" --out "$OUT" --epochs "$EPOCHS"
