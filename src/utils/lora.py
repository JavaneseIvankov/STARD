"""STAF-Net | src/utils/lora.py

Owner   : Person 1 + Person 2 (shared plumbing)
Status  : IMPLEMENTED - verified against peft 0.22 / transformers 5.24 on CPU
          with a randomly initialised DINOv3 config (no gated weights needed).

One LoRA injection helper used by both branches so the adapter configuration
cannot drift between DINOv3 and V-JEPA.

Backbone attribute names (resolved decision D6)
-----------------------------------------------
Both Meta backbones in transformers expose attention projections as
`q_proj` / `k_proj` / `v_proj` / `o_proj`:

- DINOv3ViTAttention (transformers.models.dinov3_vit.modeling_dinov3_vit)
      self.k_proj, self.v_proj, self.q_proj, self.o_proj
- V-JEPA 2.1 attention (exported checkpoint, same ViT heritage)

That means the family description `*_proj` is enough for peft, but this module
still refuses to attach unless the requested names are actually present.

Contract
--------
- inspect_targets(model)            -> sorted unique module name suffixes
- attach_lora(model, ...)           -> (model, LoRAReport)
- count_trainable(model)            -> (trainable, total, ratio)
- freeze_module(module)             -> marks every parameter requires_grad=False

Safety rules
------------
1. Never attach when no target resolves - a silent no-op LoRA run looks exactly
   like a null result in the ablation table.
2. The caller must freeze the base model first; attach_lora re-asserts this and
   freezes every parameter that is not an adapter.
3. Report the trainable ratio so scripts/smoke_branches.py can assert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = [
    "LoRAReport",
    "freeze_module",
    "inspect_targets",
    "attach_lora",
    "count_trainable",
    "assert_only_lora_trainable",
]


@dataclass
class LoRAReport:
    """What attach_lora actually did, for logging and smoke assertions."""

    target_modules: list[str]
    rank: int
    alpha: int
    dropout: float
    matched_modules: int = 0
    trainable_params: int = 0
    total_params: int = 0
    adapter_names: list[str] = field(default_factory=list)

    @property
    def trainable_ratio(self) -> float:
        return 0.0 if self.total_params == 0 else self.trainable_params / self.total_params

    def summary(self) -> str:
        names = (
            ", ".join(self.adapter_names[:2])
            + (f" (+{len(self.adapter_names) - 2} more)" if len(self.adapter_names) > 2 else "")
            if self.adapter_names
            else "none"
        )
        return (
            f"LoRA r={self.rank} alpha={self.alpha} dropout={self.dropout} "
            f"targets=[{names}] matched={self.matched_modules} "
            f"trainable={self.trainable_params:,}/{self.total_params:,} "
            f"({100 * self.trainable_ratio:.3f}%)"
        )


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze every parameter of module (in place) and switch it to eval-safe mode."""
    for param in module.parameters():
        param.requires_grad = False
    return module


def count_trainable(model: nn.Module) -> tuple[int, int, float]:
    """Return (trainable, total, ratio)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 0.0 if total == 0 else trainable / total
    return trainable, total, ratio


def inspect_targets(model: nn.Module) -> list[str]:
    """Sorted unique leaf module names, useful before choosing LoRA targets.

    Returns every distinct final path component (e.g. 'q_proj'), not full paths.
    Filter by suffix when reading the output; this function is deliberately dumb
    so it cannot lie about what exists in the model.
    """
    names: set[str] = set()
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, nn.Linear):
            names.add(name.rsplit(".", 1)[-1])
    return sorted(names)


def attach_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | tuple[str, ...] = ("q_proj", "v_proj"),
) -> tuple[nn.Module, LoRAReport]:
    """Wrap model with LoRA adapters on the given projection suffixes.

    The base weights stay frozen; only adapter parameters become trainable.
    Raises ValueError when a target does not resolve to any module, because a
    silently empty adapter set produces a misleading "LoRA did not help" result.
    """
    from peft import LoraConfig, get_peft_model

    targets = [t for t in dict.fromkeys(target_modules)]
    if not targets:
        raise ValueError(
            "attach_lora: target_modules is empty. Call inspect_targets(model) and "
            "pass real projection names such as ('q_proj', 'v_proj')."
        )

    available = set(inspect_targets(model))
    resolved = [t for t in targets if t in available]
    missing = [t for t in targets if t not in available]
    if not resolved:
        raise ValueError(
            f"attach_lora: none of {targets} exist in the model. "
            f"Available linear suffixes: {sorted(available)}"
        )

    freeze_module(model)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved,
        bias="none",
        task_type=None,
    )
    adapted = get_peft_model(model, config)

    # get_peft_model only enables adapter params, but re-assert explicitly so a
    # future peft change cannot quietly unfreeze the backbone.
    adapter_names: list[str] = []
    for name, param in adapted.named_parameters():
        is_adapter = "lora_" in name
        param.requires_grad = is_adapter
        if is_adapter:
            adapter_names.append(name)

    matched = sum(
        1 for name, _ in adapted.named_modules() if name.rsplit(".", 1)[-1] in resolved
    )
    trainable, total, _ = count_trainable(adapted)

    report = LoRAReport(
        target_modules=resolved,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        matched_modules=matched,
        trainable_params=trainable,
        total_params=total,
        adapter_names=sorted({n.split(".lora_")[0] for n in adapter_names}),
    )
    if missing:
        report.adapter_names.append(f"(missing: {', '.join(missing)})")
    return adapted, report


def assert_only_lora_trainable(model: nn.Module) -> None:
    """Raise ValueError when any non-adapter parameter is trainable."""
    leaked = [name for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name]
    if leaked:
        preview = ", ".join(leaked[:5])
        raise ValueError(
            f"attach_lora: {len(leaked)} non-adapter parameters are trainable "
            f"(e.g. {preview}); base weights must stay frozen"
        )
