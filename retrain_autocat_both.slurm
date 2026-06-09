#!/bin/bash
#SBATCH --job-name=autocat_both
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-unknown}"
NODE_NAME="${SLURMD_NODENAME:-${SLURM_NODELIST:-unknown}}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

echo "Starting job $JOB_ID on node $NODE_NAME at $(date)"

cd "$SUBMIT_DIR"
mkdir -p logs models/saved

if [ ! -d "$SUBMIT_DIR/venv" ]; then
  echo "Missing virtualenv at $SUBMIT_DIR/venv"
  exit 1
fi

source "$SUBMIT_DIR/venv/bin/activate"
pip install -e . --quiet
pip install -q "setfit>=1.1" "sentence-transformers>=3.0,<5.0" "transformers>=4.41.0,<5.0" "datasets>=2.18"

HF_DATASETS="DoDataThings/us-bank-transaction-categories,rajeshradhakrishnan/fin-transaction-category"
GOLD_CSV="evaluation/fixtures/transaction_category_regression.csv"
HF_CAP="1200"

COMMON_ARGS=(
  --from-db
  --gold-csv "$GOLD_CSV"
  --hf-dataset "$HF_DATASETS"
  --hf-max-per-category "$HF_CAP"
)

echo "=== Environment ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "HF datasets: $HF_DATASETS"
echo "Gold CSV: $GOLD_CSV"

echo ""
echo "=== Train TF-IDF classifier ==="
python training/train_description_classifier.py "${COMMON_ARGS[@]}"

echo ""
echo "=== Evaluate TF-IDF classifier ==="
python evaluation/evaluate_autocategorization.py --model models/saved/description_classifier.pkl

echo ""
echo "=== Train SetFit classifier ==="
python training/train_description_classifier.py --setfit "${COMMON_ARGS[@]}"

echo ""
echo "=== Evaluate SetFit classifier ==="
python evaluation/evaluate_autocategorization.py --model models/saved/setfit_description_classifier

echo ""
echo "=== Done ==="
echo "Finished at $(date)"
ls -lh models/saved/description_classifier.pkl || true
ls -ld models/saved/setfit_description_classifier || true
