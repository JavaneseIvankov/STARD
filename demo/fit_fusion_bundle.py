"""Fit and save the compact Stage II Standardized Fusion bundle (demo/fusion_bundle.pt)
from cached train/val features so the live inference demo uses the exact verified
0.9030 Test AUROC standardized dual-branch model + single-branch probes.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.video_dataset import read_manifest


def load_variant(features_dir: Path, variant: str) -> dict[str, np.ndarray]:
    store = {}
    for sp in ("train", "val", "test"):
        for pt in sorted((features_dir / variant / sp).glob("shard-*.pt")):
            payload = torch.load(pt, map_location="cpu", weights_only=False)
            for rec in payload["records"]:
                store[rec["video_id"]] = rec["z"].float().numpy()
    return store


def fit_branch_head(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, Xte: np.ndarray, yte: np.ndarray, C: float = 0.05) -> dict:
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42).fit(Xtr, ytr)
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    thrs = np.linspace(0.10, 0.90, 161)
    best_thr = float(thrs[np.argmax([f1_score(yva, (p_va >= t).astype(int)) for t in thrs])])
    return {
        "coef": clf.coef_[0].astype(np.float32),
        "intercept": float(clf.intercept_[0]),
        "threshold": best_thr,
        "val_auroc": float(roc_auc_score(yva, p_va)),
        "test_auroc": float(roc_auc_score(yte, p_te)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    rows = read_manifest(root / "data" / "manifests" / "manifest.csv")
    fd = load_variant(root / "data" / "features", "dino_lora")
    fj = load_variant(root / "data" / "features", "vjepa_lora")

    def get_sp(sp: str):
        sub = [r for r in rows if r.split == sp]
        xd = np.stack([fd[r.video_id] for r in sub]).astype(np.float32)
        xj = np.stack([fj[r.video_id] for r in sub]).astype(np.float32)
        y = np.array([int(r.label) for r in sub], dtype=np.int64)
        return xd, xj, y

    tr_d, tr_j, tr_y = get_sp("train")
    va_d, va_j, va_y = get_sp("val")
    te_d, te_j, te_y = get_sp("test")

    sc_d = StandardScaler().fit(tr_d)
    sc_j = StandardScaler().fit(tr_j)

    tr_ds, va_ds, te_ds = sc_d.transform(tr_d), sc_d.transform(va_d), sc_d.transform(te_d)
    tr_js, va_js, te_js = sc_j.transform(tr_j), sc_j.transform(va_j), sc_j.transform(te_j)

    dino_head = fit_branch_head(tr_ds, tr_y, va_ds, va_y, te_ds, te_y, C=0.05)
    vjepa_head = fit_branch_head(tr_js, tr_y, va_js, va_y, te_js, te_y, C=0.05)
    fusion_head = fit_branch_head(
        np.hstack([tr_ds, tr_js]), tr_y,
        np.hstack([va_ds, va_js]), va_y,
        np.hstack([te_ds, te_js]), te_y,
        C=0.05,
    )

    bundle = {
        "dino_mean": sc_d.mean_.astype(np.float32),
        "dino_scale": sc_d.scale_.astype(np.float32),
        "vjepa_mean": sc_j.mean_.astype(np.float32),
        "vjepa_scale": sc_j.scale_.astype(np.float32),
        "dino_head": dino_head,
        "vjepa_head": vjepa_head,
        "fusion_head": fusion_head,
    }
    out_path = root / "demo" / "fusion_bundle.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    for k in ("dino_head", "vjepa_head", "fusion_head"):
        h = bundle[k]
        print(f"  {k:12s}: Val AUROC={h['val_auroc']:.4f} | Test AUROC={h['test_auroc']:.4f} | thr={h['threshold']:.3f}")


if __name__ == "__main__":
    main()
