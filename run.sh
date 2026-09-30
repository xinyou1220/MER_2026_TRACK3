#!/usr/bin/env bash
# Reproduce one run end to end: data splits, training, held-out evaluation and the official submission.
#   EXPERIMENT=sympo SEED=42 bash run.sh
set -euo pipefail
EXPERIMENT="${EXPERIMENT:-sympo}"
SEED="${SEED:-42}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

[ -f splits/train.jsonl ] || { python scripts/split_weak_data.py && python scripts/build_splits.py; }
python scripts/run_experiment.py --experiment "$EXPERIMENT" --seed "$SEED"

CHECKPOINT="runs/$EXPERIMENT/seed$SEED/best_trainable.pt"
if [ -f "$CHECKPOINT" ]; then
  python scripts/infer_official.py --checkpoint "$CHECKPOINT" --out "outputs/${EXPERIMENT}_seed${SEED}.csv"
  python scripts/package_submission.py --input "outputs/${EXPERIMENT}_seed${SEED}.csv" --out "outputs/${EXPERIMENT}_seed${SEED}.zip"
fi
