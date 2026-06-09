#!/bin/bash
#SBATCH --job-name=budgetml_train
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo  "Starting job $SLURM_JOB_ID on node $SLURM_NODENAME at $(date)"


cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/saved
set -e
source "$SLURM_SUBMIT_DIR/venv/bin/activate"
pip install -e . --quiet


# Load HF_TOKEN from .env file or environment — never hardcode tokens
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. PersonaLedger download may fail."
fi
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODENAME | Start: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Step 1/9: Description Classifier — TF-IDF (CPU) ==="
python -m training.train_description_classifier

echo "=== Step 1b/9: Description Classifier — SetFit (CPU, fine-tuned on DB labels) ==="
python -c "import setfit" 2>/dev/null || pip install -q "setfit>=1.1" "sentence-transformers>=3.0,<5.0" "transformers>=4.41.0,<5.0" "datasets>=2.18"
python -m training.train_description_classifier --setfit --from-db

echo "=== Step 2/9: Masked Pretraining ==="
python -m training.pretrain_masked \
    --source db+auto --max-rows 500000 \
    --epochs 20 --batch-size 256 --lr 1e-3

echo "=== Step 3/9: Merchant2Vec ==="
python -m training.train_merchant2vec \
    --source db+auto --max-rows 500000 \
    --epochs 10 --batch-size 512 --lr 1e-3

echo "=== Step 4/9: Baselines ==="
python -m training.train_baselines --source db+auto --max-rows 500000

echo "=== Step 5/9: SpendingForecastTransformer ==="
python -m training.train_forecast \
    --source db+auto --max-rows 500000 \
    --seq-len 60 --horizon 30 \
    --d-model 256 --nhead 8 --num-layers 4 \
    --epochs 50 --batch-size 256 --lr 5e-4 --patience 7

echo "=== Step 6/9: AnomalyDetectionTransformer ==="
python -m training.train_anomaly \
    --source db+auto --max-rows 500000 \
    --seq-len 60 --d-model 256 --nhead 8 --num-layers 4 \
    --lambda-cls 0.5 --epochs 100 --batch-size 256 --lr 5e-4 --patience 10

echo "=== Step 7/9: ContrastiveAnomalyDetector ==="
python -m training.train_contrastive_anomaly \
    --source db+auto --max-rows 500000 \
    --epochs 20 --batch-size 256 --lr 1e-3

echo "=== Step 8/9: BulkBuyRecommendationTransformer ==="
python -m training.train_bulk_buy \
    --source db+auto --max-rows 500000 \
    --seq-len 60 --d-model 256 --nhead 8 --num-layers 4 \
    --epochs 50 --batch-size 256 --lr 5e-4 --patience 7

echo "=== Step 9/9: UserSpendingVAE + Cohort Stats ==="
python -m training.train_user_vae \
    --source db+auto --max-rows 500000 \
    --epochs 20 --batch-size 256 --lr 5e-4 --beta 1.0

echo "=== Final Evaluation ==="
python -m evaluation.evaluate --source db+auto --max-rows 500000

echo "Done: $(date)"
echo "Time taken: $(($SECONDS / 3600))h $((($SECONDS % 3600) / 60))m $(($SECONDS % 60))s"
ls -lh models/saved/
