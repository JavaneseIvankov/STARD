# Open decisions (blocking parallel work)

Resolved items moved to the table at the bottom. Only unresolved decisions stay
at the top, each with the module it blocks and a recommendation.

---

## D3 — Does V-JEPA 2.1 accept 16 frames at 384?

Blocks: `src/models/vjepa_branch.py` (Person 2), and every A2/A4/A5-A8 number

The checkpoint declares `crop_size=384`, `frames_per_clip=64`, `tubelet_size=2`,
`interpolate_rope=true`. This project feeds 16 frames at 384.

- Temporal: 16 frames / tubelet 2 = `T_out = 8` before pooling.
- Spatial: 384 input matches the pretrained crop, so RoPE spatial interpolation
  is not required here (unlike a 224 input).

Recommendation: load the released encoder, feed a real 16-frame clip, and record
the output structure before writing LoRA. If 16 frames degrades the frozen
baseline, compare against 32 frames on a small validation slice and document the
choice. Do not assume; the whole A2/A4 baseline depends on it.

Owner: Person 2. Deadline: before Person 3 trains A2/A4.

---

## D5 — What exactly is pooled?

Blocks: `src/models/dino_branch.py`, `src/models/vjepa_branch.py`

`src/contracts.py` exposes `POOLING = "mean"`, but the precise definition must be
identical across branches and across frozen/LoRA runs, otherwise A1 vs A3 is not
a controlled comparison.

Candidates:

- DINO: mean over the 4 frame CLS tokens, mean over patch tokens, or concat of both.
- V-JEPA: mean over the token grid returned by the video encoder.

Recommendation: fix one definition in writing, implement it in a single shared
function, and assert the output width against the contract. Prefer CLS for DINO
(image-level semantics) and mean over tokens for V-JEPA (no single CLS token is
guaranteed), then note the asymmetry in the report.

Owner: Person 1 + Person 2, agreed before either branch trains.

---

## D6 — LoRA target module names

Blocks: `src/utils/lora.py`, both branch trainings

Both the plan and the guide say "verify module names first" and stop there. Wrong
targets produce a LoRA run that trains nothing and looks like a null result.

Recommendation: `src/utils/lora.py` ships `inspect_targets(model)` that prints
sorted attention projection names and refuses to attach when `target_modules` is
empty. Record the chosen names per backbone in the branch docstring.

Owner: Person 1 (DINOv3) + Person 2 (V-JEPA).

---

## D4 — Licensing of frozen backbone weights

Blocks: paper release, checkpoint distribution

- DINOv3 `pretrained_cfg.license` is `dinov3`, not MIT. Gated HF repo.
- V-JEPA 2.1 ships under Meta's own terms.

`LICENSE` (MIT) covers our code only. Third-party weights and derived cached
features may carry restrictions.

Recommendation: keep MIT for code, add `THIRD_PARTY_WEIGHTS.md` recording each
backbone's license and access terms, and do not redistribute `.pt` caches or
checkpoints publicly until confirmed.

Owner: whoever writes the paper.

---

## D7 — Compute budget vs ablation matrix

Blocks: the 4-hour A100 schedule

Stage I LoRA on V-JEPA (16 frames, 384, batch 1-2) can consume most of the
budget. The plan already defines a fallback order; it needs a measured trigger,
not a feeling.

Recommendation: after 30-50 representative optimizer steps, compute
`T_epoch = ceil(N_train / B_eff) * T_step` and publish the number in the report.
If projected Stage I time exceeds 2 h, drop A3/A4 first, keep A1/A2/A5/A6, and
state the fallback explicitly.

Owner: Person 2 publishes the measurement, team decides.

---

## Resolved

| ID | Decision | Date |
| :--- | :--- | :--- |
| — | Task is binary real/fake on DFDC; 3-class + physical spoof dropped | 2026-02 |
| D1 | No spatial/temporal masks. DFDC has video-level labels only; classification-only pipeline | 2026-02 |
| D2 | Splits are assigned per source group derived from DFDC `original` linkage, not per video | 2026-02 |
| — | Contract v0.2: DINO 4×224×384 `(B,T,C,H,W)`, V-JEPA 16×384×768 `(B,C,T,H,W)`, projection 256 | 2026-02 |
| — | Two-stage training: Stage I branch frozen/LoRA, Stage II cached-feature fusion and A1-A8 | 2026-02 |
| — | Feature cache stores fp32 pooled vectors per video, 4 variants (`docs/cache_format.md`) | 2026-02 |
