"""STAF-Net | src/models/stafnet.py

Owner   : Person 3 (Fusion / evaluation)
Status  : IMPLEMENTED - supports A1-A8 ablations (single_dino, single_vjepa,
          concat, gated) under src.contracts.FusionInput -> STAFNetOutput.

Formulas (plan §7.1-§7.3)
-------------------------
    h_D = GELU(W_D z_D + b_D)          z_D (B, 384) -> (B, 256)
    h_J = GELU(W_J z_J + b_J)          z_J (B, 768) -> (B, 256)
    concat : logit = MLP([h_D || h_J])                 (512 -> 128 -> 1)
    gated  : g = sigmoid(W_g [h_D || h_J] + b_g)       (512 -> 256)
             h_fused = g * h_D + (1 - g) * h_J
             logit = MLP(h_fused)                      (256 -> 128 -> 1)
    single : logit = MLP(h_D) or MLP(h_J)              (256 -> 128 -> 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.contracts import (
    DINO_DIM,
    DROPOUT,
    D_PROJ,
    HEAD_HIDDEN,
    NUM_LOGITS,
    VJEPA_DIM,
    FusionInput,
    STAFNetOutput,
)

__all__ = ["STAFNet", "FUSION_MODES"]

FUSION_MODES: tuple[str, ...] = ("single_dino", "single_vjepa", "concat", "gated")


class STAFNet(nn.Module):
    """Unified Stage II classifier & fusion module for ablations A1-A8."""

    def __init__(
        self,
        mode: str = "gated",
        *,
        dino_dim: int = DINO_DIM,
        vjepa_dim: int = VJEPA_DIM,
        d_proj: int = D_PROJ,
        hidden_dim: int = HEAD_HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"STAFNet: unknown mode {mode!r}, expected one of {FUSION_MODES}")
        self.mode = mode
        self.dino_dim = dino_dim
        self.vjepa_dim = vjepa_dim
        self.d_proj = d_proj
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.proj_dino: nn.Module | None = None
        self.proj_vjepa: nn.Module | None = None
        self.gate_layer: nn.Module | None = None

        if mode in ("single_dino", "concat", "gated"):
            self.proj_dino = nn.Sequential(
                nn.Linear(dino_dim, d_proj),
                nn.GELU(),
            )
        if mode in ("single_vjepa", "concat", "gated"):
            self.proj_vjepa = nn.Sequential(
                nn.Linear(vjepa_dim, d_proj),
                nn.GELU(),
            )

        if mode == "gated":
            self.gate_layer = nn.Linear(d_proj * 2, d_proj)
            head_in = d_proj
        elif mode == "concat":
            head_in = d_proj * 2
        else:
            head_in = d_proj

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_LOGITS),
        )

    def forward(self, batch: FusionInput) -> STAFNetOutput:
        gate: torch.Tensor | None = None

        if self.mode == "single_dino":
            if batch.dino is None or self.proj_dino is None:
                raise ValueError("STAFNet[single_dino]: batch.dino is required")
            fused = self.proj_dino(batch.dino)
        elif self.mode == "single_vjepa":
            if batch.vjepa is None or self.proj_vjepa is None:
                raise ValueError("STAFNet[single_vjepa]: batch.vjepa is required")
            fused = self.proj_vjepa(batch.vjepa)
        else:
            if batch.dino is None or batch.vjepa is None:
                raise ValueError(f"STAFNet[{self.mode}]: both batch.dino and batch.vjepa are required")
            assert self.proj_dino is not None and self.proj_vjepa is not None
            h_d = self.proj_dino(batch.dino)
            h_j = self.proj_vjepa(batch.vjepa)
            h_cat = torch.cat([h_d, h_j], dim=-1)
            if self.mode == "concat":
                fused = h_cat
            else:
                assert self.gate_layer is not None
                gate = torch.sigmoid(self.gate_layer(h_cat))
                fused = gate * h_d + (1.0 - gate) * h_j

        logit = self.classifier(fused)
        return STAFNetOutput(logit=logit, gate=gate)
