"""
training/train_merchant2vec.py — Train Merchant2Vec skip-gram embeddings.

Treats each user's transaction history (sorted by date) as a sentence of merchant
"words" and trains a skip-gram model with negative sampling. Context window = 5.

Output
------
    models/saved/merchant2vec.pt   — Merchant2Vec state dict + metadata

Usage
-----
    python training/train_merchant2vec.py --source personaledger --epochs 10
    python training/train_merchant2vec.py --source synthetic --epochs 2  # smoke test
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import joblib

from config import Settings
from data.loader import load_transactions
from data.preprocessor import TransactionPreprocessor
from models.embeddings.merchant2vec import Merchant2Vec


class MerchantSkipgramDataset(Dataset):
    """
    Generates (center, context, negatives) triples from merchant sequences.

    Parameters
    ----------
    sequences      : list of lists — each inner list is a user's ordered merchant ids
    vocab_size     : int
    window         : int   context window radius (default 5)
    n_negatives    : int   negative samples per positive (default 5)
    freq_weights   : np.ndarray of shape (vocab_size,) for negative sampling
    """

    def __init__(
        self,
        sequences:   list[list[int]],
        vocab_size:  int,
        window:      int         = 5,
        n_negatives: int         = 5,
        freq_weights: np.ndarray | None = None,
    ):
        self.n_negatives = n_negatives
        self.vocab_size  = vocab_size

        # Negative sampling distribution: frequency^(3/4) (Mikolov et al.)
        if freq_weights is not None:
            w = freq_weights ** 0.75
        else:
            w = np.ones(vocab_size)
        self.neg_probs = w / w.sum()

        # Build all (center, context) pairs
        centers, contexts = [], []
        for seq in sequences:
            for i, center in enumerate(seq):
                lo = max(0, i - window)
                hi = min(len(seq), i + window + 1)
                for j in range(lo, hi):
                    if j != i:
                        centers.append(center)
                        contexts.append(seq[j])

        self.centers  = torch.tensor(centers,  dtype=torch.long)
        self.contexts = torch.tensor(contexts, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int) -> tuple:
        center  = self.centers[idx]
        context = self.contexts[idx]
        negs    = torch.from_numpy(
            np.random.choice(self.vocab_size, self.n_negatives, p=self.neg_probs)
        ).long()
        return center, context, negs


def train(args) -> None:
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data (source={args.source}) …")
    pl_cfg = cfg.dataset("personaledger")
    df = load_transactions(
        source=args.source,
        personaledger_config=pl_cfg.default_hf_config,
        personaledger_max_rows=args.max_rows or pl_cfg.default_max_rows,
    )
    print(f"  {len(df):,} transactions, {df['user_id'].nunique()} users")

    # ── Load or fit preprocessor ──────────────────────────────────────────────
    prep_path = cfg.save_dir / "preprocessor.pkl"
    if prep_path.exists():
        prep = joblib.load(prep_path)
    else:
        pcfg = cfg.preprocessor
        prep = TransactionPreprocessor(
            seq_len=pcfg.seq_len,
            forecast_horizon=pcfg.forecast_horizon,
            max_merchants=pcfg.max_merchants,
        )
        all_users = df["user_id"].unique()
        np.random.default_rng(42).shuffle(all_users)
        prep.fit(df[df["user_id"].isin(all_users[:int(0.8 * len(all_users))])])
        joblib.dump(prep, prep_path)

    vocab_size = len(prep.merch_encoder.classes_)
    print(f"  Merchant vocabulary size: {vocab_size}")

    # ── Build merchant sequences per user ─────────────────────────────────────
    sequences: list[list[int]] = []
    df_sorted = df.sort_values(["user_id", "date"])
    merch_map = prep._merch_map
    merch_set = prep._merch_set
    unk_id    = merch_map.get("<UNK>", 0)

    for uid, udf in df_sorted.groupby("user_id"):
        seq = [
            merch_map.get(m, unk_id) if m in merch_set else unk_id
            for m in udf["merchant"].values
        ]
        if len(seq) >= 2:
            sequences.append(seq)

    print(f"  Built {len(sequences)} user sequences")

    # ── Frequency weights for negative sampling ───────────────────────────────
    all_ids = [mid for seq in sequences for mid in seq]
    counts  = Counter(all_ids)
    freq    = np.array([counts.get(i, 1) for i in range(vocab_size)], dtype=np.float32)

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    dataset = MerchantSkipgramDataset(
        sequences=sequences,
        vocab_size=vocab_size,
        window=5,
        n_negatives=5,
        freq_weights=freq,
    )
    print(f"  Skip-gram pairs: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or 4096,
        shuffle=True,
        num_workers=min(4, __import__("os").cpu_count() or 1),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = Merchant2Vec(vocab_size=vocab_size, emb_dim=32)
    # Sparse AdaGrad is standard for skip-gram (fast sparse updates)
    optimizer = optim.SparseAdam(list(model.parameters()), lr=args.lr or 0.01)

    epochs = args.epochs or 10

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        total_loss, total_n = 0.0, 0
        t0 = time.time()
        model.train()
        for center, context, negatives in loader:
            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            n = len(center)
            total_loss += loss.item() * n
            total_n    += n
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={total_loss / max(total_n, 1):.4f}  {elapsed:.1f}s"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = cfg.save_dir / "merchant2vec.pt"
    model.save(save_path)
    print(f"\nSaved → {save_path}")
    print(f"  Embedding matrix shape: {model.get_embeddings().shape}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Merchant2Vec skip-gram embeddings")
    p.add_argument("--source",     default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--max-rows",   type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None,
                   help="Training epochs (default 10)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Mini-batch size for skip-gram training (default 4096)")
    p.add_argument("--lr",         type=float, default=None,
                   help="SparseAdam learning rate (default 0.01)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
