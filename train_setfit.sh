#!/usr/bin/env bash
# train_setfit.sh — Fine-tune the SetFit description classifier.
#
# Usage:
#   ./train_setfit.sh                  # seed data only
#   ./train_setfit.sh --from-db        # seed + your labeled DB transactions
#   ./train_setfit.sh --iterations 40  # more contrastive pairs (slower, more accurate)
#
# Runs on CPU — no GPU allocation needed. Takes ~5 min on a login node.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/venv/bin/activate"

echo "=== Checking SetFit deps ==="
pip install -q "setfit>=1.1" "sentence-transformers>=3.0,<5.0" "transformers>=4.41.0,<5.0" "datasets>=2.18"

echo ""
echo "=== Training SetFit description classifier ==="
python -m training.train_description_classifier --setfit "$@"

echo ""
echo "=== Done. Model saved to models/saved/setfit_description_classifier/ ==="
echo "Restart the API to load the new model:"
echo "  uvicorn api.app:app --reload --port 8000"
