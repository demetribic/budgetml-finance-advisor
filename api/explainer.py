"""
api/explainer.py — Attention-based and feature-sensitivity explainability.

Provides two explanation modes for BudgetML model outputs:
  1. Attention-based  : extract which time steps the [CLS] token attended to most
  2. Feature-based    : perturb each feature ±10% and rank by sensitivity (SHAP lite)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from models.transformers.spending_forecast import SpendingForecastTransformer
    from models.transformers.anomaly_detection import AnomalyDetectionTransformer


# Feature names in order (matches preprocessor feature vector)
FEATURE_NAMES = [
    "amount_norm",
    "cat_id",
    "merch_id",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
    "month_sin",
    "month_cos",
    "user_amount_zscore",
    "description_hash",
]


class SuggestionExplainer:
    """
    Attaches evidence to each suggestion explaining what triggered it.

    Two explanation modes:
    1. Attention-based: forward hook on the last TransformerEncoderLayer,
       extract CLS → token attention weights, report most-attended time step.
    2. Feature-based: perturb each feature by ±10%, measure delta in output,
       rank features by mean absolute sensitivity.
    """

    _attn_lock = threading.Lock()  # serialize monkey-patch across concurrent requests

    def explain_forecast(
        self,
        model:        "SpendingForecastTransformer",
        x:            Tensor,
        category_idx: int,
    ) -> dict:
        """
        Explain which time steps and features drove a forecast prediction.

        Parameters
        ----------
        model        : SpendingForecastTransformer
        x            : (1, seq_len, feature_dim)  — single sample batch
        category_idx : int   which output category to explain (0–8)

        Returns
        -------
        dict:
          most_influential_timestep : int
          attention_weights         : list[float] (seq_len,)
          feature_sensitivity       : dict[str, float]
          explanation_text          : str
        """
        attn_weights = self._extract_attention(model.encoder, x)    # (seq_len,)
        top_step     = int(attn_weights.argmax())

        feat_sens = self._feature_sensitivity(
            lambda inp: model(inp)[:, category_idx],
            x,
        )

        # Top 3 most sensitive features
        top_feats = sorted(feat_sens.items(), key=lambda kv: kv[1], reverse=True)[:3]
        feat_str  = ", ".join(f"{k} ({v:.3f})" for k, v in top_feats)
        attn_pct  = float(attn_weights[top_step]) * 100

        explanation_text = (
            f"The forecast is most influenced by time step {top_step} "
            f"({attn_pct:.1f}% of attention). "
            f"Top driving features: {feat_str}."
        )

        return {
            "most_influential_timestep": top_step,
            "attention_weights":         attn_weights.tolist(),
            "feature_sensitivity":       feat_sens,
            "explanation_text":          explanation_text,
        }

    def explain_anomaly(
        self,
        model: "AnomalyDetectionTransformer",
        x:     Tensor,
    ) -> dict:
        """
        Explain which time step and features contributed most to anomaly score.

        Returns the step with the highest per-step reconstruction error and
        which features were reconstructed worst at that step.

        Parameters
        ----------
        model : AnomalyDetectionTransformer
        x     : (1, seq_len, feature_dim)

        Returns
        -------
        dict:
          most_anomalous_timestep   : int
          per_step_error            : list[float]
          feature_reconstruction_error : dict[str, float]
          explanation_text          : str
        """
        model.eval()
        with torch.no_grad():
            out = model(x)
        recon = out["recon"]    # (1, seq_len, feature_dim)

        # Per-step MSE
        per_step_error = F.mse_loss(recon, x, reduction="none").mean(dim=-1)[0]  # (seq_len,)
        top_step       = int(per_step_error.argmax())
        step_error_list = per_step_error.tolist()

        # Per-feature reconstruction error at the most anomalous step
        step_recon   = recon[0, top_step, :]    # (feature_dim,)
        step_original = x[0, top_step, :]
        feat_errors  = (step_recon - step_original).abs().tolist()
        feat_dim     = min(len(feat_errors), len(FEATURE_NAMES))
        feat_err_dict = {
            FEATURE_NAMES[i]: round(float(feat_errors[i]), 4)
            for i in range(feat_dim)
        }

        top_feat = max(feat_err_dict, key=feat_err_dict.get)
        explanation_text = (
            f"The anomaly is concentrated at time step {top_step} "
            f"(reconstruction error: {step_error_list[top_step]:.4f}). "
            f"The most poorly reconstructed feature is '{top_feat}'."
        )

        return {
            "most_anomalous_timestep":        top_step,
            "per_step_error":                 [round(e, 4) for e in step_error_list],
            "feature_reconstruction_error":   feat_err_dict,
            "explanation_text":               explanation_text,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_attention(
        self,
        encoder,
        x: Tensor,
    ) -> Tensor:
        """
        Register a forward hook on the last TransformerEncoderLayer to capture
        the multi-head attention weights. Average over heads, return the row
        corresponding to the [CLS] token (position 0) for positions 1..T+1.

        Returns (seq_len,) attention weight tensor (sums to 1).
        """
        attn_output: list[Tensor] = []

        def _hook(module, inp, output):
            # output[1] is the attention weight tensor when need_weights=True
            # Shape: (batch, nhead, T+1, T+1) or (batch, T+1, T+1)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_output.append(output[1].detach())

        # Patch the last encoder layer's self_attn to return weights
        last_layer = encoder.encoder.layers[-1]
        original_forward = last_layer.self_attn.forward

        def patched_attn(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            return original_forward(*args, **kwargs)

        with self._attn_lock:
            last_layer.self_attn.forward = patched_attn
            handle = last_layer.register_forward_hook(_hook)
            try:
                encoder.eval()
                with torch.no_grad():
                    encoder(x)
            finally:
                handle.remove()
                last_layer.self_attn.forward = original_forward

        if not attn_output:
            seq_len = x.size(1)
            return torch.ones(seq_len) / seq_len

        weights = attn_output[0]   # (batch, T+1, T+1) averaged over heads
        if weights.dim() == 4:
            weights = weights.mean(dim=1)
        # CLS row (index 0) → attention over positions 1..T (the actual sequence)
        cls_attn = weights[0, 0, 1:]     # (seq_len,)
        # Renormalize to sum 1
        cls_attn = cls_attn / (cls_attn.sum() + 1e-8)
        return cls_attn

    def _feature_sensitivity(
        self,
        score_fn,           # Callable: Tensor → Tensor scalar
        x: Tensor,
        delta: float = 0.10,
    ) -> dict[str, float]:
        """
        Cheap SHAP approximation: perturb each feature by ±delta, measure output
        change, return mean absolute sensitivity per feature.

        Returns dict {feature_name: sensitivity_score}.
        """
        feat_dim = x.size(-1)
        with torch.no_grad():
            base = float(score_fn(x).mean())

        sensitivities: dict[str, float] = {}
        for i in range(feat_dim):
            x_plus  = x.clone()
            x_minus = x.clone()
            x_plus[:, :, i]  *= (1 + delta)
            x_minus[:, :, i] *= (1 - delta)

            with torch.no_grad():
                s_plus  = float(score_fn(x_plus).mean())
                s_minus = float(score_fn(x_minus).mean())

            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
            sensitivities[name] = round(abs(s_plus - base) + abs(s_minus - base), 5)

        return sensitivities
