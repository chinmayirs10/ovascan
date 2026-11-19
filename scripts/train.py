"""Fine-tune MobileNetV3-Small on the leakage-free split.

Usage:
    python scripts/train.py --epochs 8

Reads splits/split.csv (from audit_dataset.py), writes:
    models/pcos_mobilenetv3.pt
    reports/metrics.json
    reports/training_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcos_screen.data import read_split_csv  # noqa: E402
from pcos_screen.model import HIGH, LOW, build_model, eval_transform, train_transform  # noqa: E402


class SplitDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(ROOT / r["path"]).convert("RGB")
        return self.transform(img), torch.tensor(float(r["label"]))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x).squeeze(1)))
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy().astype(int)


def metrics(probs, y, threshold=0.5):
    eps = 1e-7
    bce = float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    auc = roc_auc_score(y, probs) if len(set(y)) == 2 else float("nan")
    inconclusive = float(((probs >= LOW) & (probs <= HIGH)).mean())
    return {
        "n": int(len(y)), "bce_loss": bce, "accuracy": float((pred == y).mean()),
        "sensitivity": float(sens), "specificity": float(spec), "roc_auc": float(auc),
        "confusion_matrix [[tn, fp], [fn, tp]]": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "share_inconclusive_band": inconclusive,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training images")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = read_split_csv(ROOT / "splits" / "split.csv")
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if args.limit:
        by = {s: v[: args.limit] for s, v in by.items()}
    print({s: len(v) for s, v in by.items()})

    # balance classes in the loss
    n_pos = sum(int(r["label"]) for r in by["train"])
    pos_weight = torch.tensor((len(by["train"]) - n_pos) / max(n_pos, 1))

    loaders = {
        "train": DataLoader(SplitDataset(by["train"], train_transform), batch_size=args.batch, shuffle=True),
        "val": DataLoader(SplitDataset(by["val"], eval_transform), batch_size=args.batch),
        "test": DataLoader(SplitDataset(by["test"], eval_transform), batch_size=args.batch),
    }

    model = build_model(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # select on validation loss: AUC saturates at 1.0 quickly and says nothing about calibration
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(by["train"])
        vp, vy = predict(model, loaders["val"])
        vm = metrics(vp, vy)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in vm.items()
                                                                     if not k.startswith("confusion")}})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={vm['bce_loss']:.4f} "
              f"val_auc={vm['roc_auc']:.3f} val_acc={vm['accuracy']:.3f}")
        if vm["bce_loss"] < best_loss:
            best_loss, best_state = vm["bce_loss"], {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tp, ty = predict(model, loaders["test"])
    test_m = metrics(tp, ty)
    print("TEST:", json.dumps(test_m, indent=2))

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / "pcos_mobilenetv3.pt")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps({
        "split": "family-grouped (see reports/data_audit.md)",
        "split_sizes": {s: len(v) for s, v in by.items()},
        "best_val_loss": best_loss, "test": test_m, "history": history, "args": vars(args),
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o")
    ax[0].set_title("Train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["val_roc_auc"] for h in history], marker="o", label="val ROC-AUC")
    ax[1].plot(ep, [h["val_accuracy"] for h in history], marker="o", label="val accuracy")
    ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "training_curves.png", dpi=120)


if __name__ == "__main__":
    main()
