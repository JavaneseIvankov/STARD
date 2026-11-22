"""STAF-Net | src/utils/lora.py

Owner   : Person 1 + Person 2 (shared plumbing)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- One LoRA injection helper used by both branches so adapter configuration
  cannot drift between DINOv3 and V-JEPA.
- Print/inspect candidate target modules before wrapping anything.

Contract
--------
- attach_lora(model, *, rank=8, alpha=16, dropout=0.05, target_modules) -> model
- Backbone weights stay frozen; only adapter + head parameters are trainable.
- Must report trainable parameter count so the smoke test can assert it.

TODO
----
- [ ] inspect_targets(model) -> sorted list of attention projection names
- [ ] attach_lora(...) using peft LoraConfig
- [ ] count_trainable(model) helper for logging and smoke assertions
- [ ] Guard: refuse to attach when target_modules is empty
"""

# TODO(person-1,person-2): implement the shared LoRA helper described above.
