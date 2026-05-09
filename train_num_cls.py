"""Standalone source-number classification training."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import DDCRNet
from models.lossfunction import get_scheduler
from src.utils import set_unified_seed


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _parse_int_list(text: str) -> list[int]:
    vals = []
    for t in str(text).split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("M_list is empty. Provide values like '1,2,3'.")
    return sorted(set(vals))


def _build_mix_suffix(
    N: int,
    T: int,
    snr: float,
    signal_type: str,
    signal_nature: str,
    eta: float,
    bias: float,
    sv_noise_var: float,
    doa_gap: float,
    fixed_gap: bool,
    m_list: list[int],
) -> str:
    ms = "-".join(str(int(m)) for m in sorted(set(m_list)))
    fg = "fixed" if bool(fixed_gap) else "rand"
    return (
        f"_MIXM={ms}_N={int(N)}_T={int(T)}_SNR={float(snr)}"
        f"_signal={str(signal_type)}_{str(signal_nature)}"
        f"_eta={float(eta)}_bias={float(bias)}_sv_noise_var={float(sv_noise_var)}"
        f"_doa_gap={float(doa_gap)}_{fg}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train standalone source-number classifier")
    p.add_argument("--data_dir", type=str, default="datasets/generated_snapshots")

    p.add_argument("--N", type=int, default=8)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--signal_type", type=str, default="NarrowBand")
    p.add_argument("--signal_nature", type=str, default="coherent")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--sv_noise_var", type=float, default=0.0)
    p.add_argument("--doa_gap", type=float, default=10.0)
    p.add_argument("--fixed_gap", action="store_true")

    p.add_argument("--M_list", type=str, default="1,2,3")
    p.add_argument("--mix_suffix", type=str, default=None, help="Optional explicit dataset suffix")

    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--num_sched_epochs", type=int, default=80)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--base_lr", type=float, default=2.5e-4)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_confusion_matrix", action="store_true", default=True)
    return p


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def _plot_confusion_matrix(cm: np.ndarray, out_png: Path, class_names: list[str], normalize: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[num-cls][WARN] Skip plotting confusion matrix because matplotlib is unavailable: {e}")
        return

    cm_show = cm.astype(np.float64)
    if normalize:
        row_sum = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, np.maximum(row_sum, 1.0))

    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_show, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    thresh = np.nanmax(cm_show) * 0.6 if cm_show.size > 0 else 0.0
    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            if normalize:
                txt = f"{cm_show[i, j]:.2f}"
            else:
                txt = str(int(cm_show[i, j]))
            color = "white" if cm_show[i, j] > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    loss_sum = 0.0
    n = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        logits = out[0]
        loss = ce(logits, y)
        b = y.size(0)
        loss_sum += float(loss.detach().cpu()) * b
        n += b
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())

    return loss_sum / max(1, n), correct / max(1, n)


@torch.no_grad()
def _collect_preds_and_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_all = []
    y_all = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits = out[0]
        pred = torch.argmax(logits, dim=1)
        pred_all.append(pred.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    if not pred_all:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(pred_all, axis=0), np.concatenate(y_all, axis=0)


def main() -> None:
    args = _build_parser().parse_args()
    set_unified_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_list = _parse_int_list(args.M_list)
    max_m = max(m_list)

    suffix = args.mix_suffix or _build_mix_suffix(
        N=args.N,
        T=args.T,
        snr=args.snr,
        signal_type=args.signal_type,
        signal_nature=args.signal_nature,
        eta=args.eta,
        bias=args.bias,
        sv_noise_var=args.sv_noise_var,
        doa_gap=args.doa_gap,
        fixed_gap=args.fixed_gap,
        m_list=m_list,
    )

    data_dir = Path(args.data_dir)
    cov_file = data_dir / f"covariances{suffix}.npy"
    count_file = data_dir / f"source_count{suffix}.npy"
    if not cov_file.exists() or not count_file.exists():
        raise FileNotFoundError(
            f"Missing mixed-M dataset files:\n- {cov_file}\n- {count_file}\n"
            "Run f_gendata.py with --mix_M first, or provide --mix_suffix."
        )

    cov = np.load(cov_file)  # (B,T,N,N) complex
    counts = np.load(count_file).astype(np.int64)  # (B,)
    if cov.ndim != 4:
        raise ValueError(f"Expected covariances with shape (B,T,N,N), got {cov.shape}")

    # Mean over snapshots -> (B,N,N), then real/imag stack -> (B,N,N,2)
    cov_mean = cov.mean(axis=1)
    x = np.stack([cov_mean.real, cov_mean.imag], axis=-1).astype(np.float32)

    # label in [0, C-1], where class i means source count (i+1)
    y = counts - 1
    if np.min(y) < 0 or np.max(y) >= max_m:
        raise ValueError(f"Invalid source_count labels for max_m={max_m}. Found range [{y.min()}, {y.max()}]")

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    train_ratio = max(0.0, min(1.0, args.train_ratio))
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    if train_size <= 0:
        raise ValueError("train_ratio too small: no train samples")

    split_seed = int(args.split_seed)
    split_gen = None if split_seed < 0 else torch.Generator().manual_seed(split_seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if val_size > 0 else None

    model = DDCRNet(
        M=max_m,
        N=args.N,
        r=1,
        dim=args.dim,
        depth=args.depth,
        attn_dim=args.attn_dim,
        k_len=args.N,
        task_mode="num_cls",
        num_cls_max_sources=max_m,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs=args.warmup,
        total_epochs=args.num_sched_epochs,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
    )
    ce = torch.nn.CrossEntropyLoss()

    save_dir = Path("datasets") / "weights_num_cls" / f"model{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "model_numcls_best.pth"

    best_metric = -1.0
    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []

    print(f"[num-cls] device={device}, train={train_size}, val={val_size}, classes=1..{max_m}")
    print(f"[num-cls] loading data: {cov_file.name} / {count_file.name}")

    for epoch in range(args.num_epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        correct = 0

        for bi, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            logits = out[0]
            loss = ce(logits, yb)
            loss.backward()
            optimizer.step()

            b = yb.size(0)
            loss_sum += float(loss.detach().cpu()) * b
            n += b
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())

            if (bi + 1) % max(1, int(args.log_interval)) == 0 or (bi + 1) == len(train_loader):
                acc_b = float((pred == yb).float().mean().detach().cpu())
                print(
                    f"Epoch {epoch+1}/{args.num_epochs} Batch {bi+1}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu()):.4f} acc={acc_b:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        if epoch < int(args.num_sched_epochs):
            scheduler.step()

        train_loss = loss_sum / max(1, n)
        train_acc = correct / max(1, n)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, device)
        else:
            val_loss, val_acc = float("nan"), float("nan")
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        metric = val_acc if val_loader is not None else train_acc
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "suffix": suffix,
                    "max_m": max_m,
                    "m_list": m_list,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[num-cls] New best checkpoint saved: {ckpt_path} (acc={best_metric:.4f})")

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f} | time={dt:.1f}s"
        )

    np.savetxt(save_dir / "train_loss_numcls.txt", np.array(train_loss_hist), fmt="%.6f")
    np.savetxt(save_dir / "train_acc_numcls.txt", np.array(train_acc_hist), fmt="%.6f")
    np.savetxt(save_dir / "val_loss_numcls.txt", np.array(val_loss_hist), fmt="%.6f")
    np.savetxt(save_dir / "val_acc_numcls.txt", np.array(val_acc_hist), fmt="%.6f")

    if bool(getattr(args, "save_confusion_matrix", True)) and val_loader is not None and len(val_set) > 0:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        pred_np, y_np = _collect_preds_and_labels(model, val_loader, device)
        cm = _confusion_matrix(y_np, pred_np, num_classes=max_m)
        cm_norm = cm.astype(np.float64)
        cm_norm = np.divide(cm_norm, np.maximum(cm_norm.sum(axis=1, keepdims=True), 1.0))

        np.savetxt(save_dir / "val_confusion_matrix_numcls.txt", cm, fmt="%d")
        np.savetxt(save_dir / "val_confusion_matrix_numcls_norm.txt", cm_norm, fmt="%.6f")
        np.save(save_dir / "val_confusion_matrix_numcls.npy", cm)

        cls_names = [f"M={i+1}" for i in range(max_m)]
        _plot_confusion_matrix(cm, save_dir / "val_confusion_matrix_numcls.png", cls_names, normalize=False)
        _plot_confusion_matrix(cm, save_dir / "val_confusion_matrix_numcls_norm.png", cls_names, normalize=True)
        print(f"[num-cls] Confusion matrix saved to: {save_dir}")

    print("[num-cls] Training completed.")
    print(f"[num-cls] Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
