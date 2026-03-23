import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class EpochMetrics:
    train_loss: Optional[float] = None
    loss_rot: Optional[float] = None
    loss_trans: Optional[float] = None
    val_loss: Optional[float] = None
    train_t_cm: Optional[float] = None
    train_r_deg: Optional[float] = None
    val_t_cm: Optional[float] = None
    val_r_deg: Optional[float] = None


def _extract_epoch(line: str) -> Optional[int]:
    # Handles both clean lines: "Epoch 001 | ..." and noisy ones containing tqdm text.
    matches = re.findall(r"\bEpoch\s+(\d{1,6})\b", line)
    if not matches:
        return None
    return int(matches[-1])


def _extract_train_metrics(line: str) -> Optional[Tuple[float, float, float]]:
    if "Train" not in line or "Loss" not in line:
        return None

    m_loss = re.search(rf"\bLoss:\s*({_FLOAT})\b", line)
    m_rot = re.search(rf"\bLoss_rot:\s*({_FLOAT})\b", line)
    m_trans = re.search(rf"\bLoss_trans:\s*({_FLOAT})\b", line)

    if not (m_loss and m_rot and m_trans):
        return None

    return float(m_loss.group(1)), float(m_rot.group(1)), float(m_trans.group(1))


def _extract_t_r(line: str) -> Optional[Tuple[float, float]]:
    # Matches patterns like: "T: 111.10cm | R: 48.59°"
    m_t = re.search(rf"\bT:\s*({_FLOAT})\s*cm\b", line)
    # Degree symbol may be missing depending on encoding; accept optional "°".
    m_r = re.search(rf"\bR:\s*({_FLOAT})\s*(?:°)?\b", line)
    if not (m_t and m_r):
        return None
    return float(m_t.group(1)), float(m_r.group(1))


def _extract_val_loss(line: str) -> Optional[float]:
    if "Val" not in line or "Loss" not in line:
        return None

    m = re.search(rf"\bLoss:\s*({_FLOAT})\b", line)
    if not m:
        return None
    return float(m.group(1))


def parse_log(
    log_path: str,
) -> Tuple[
    List[int],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    metrics_by_epoch: Dict[int, EpochMetrics] = {}
    current_epoch: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            epoch = _extract_epoch(line)
            if epoch is not None:
                current_epoch = epoch
                metrics_by_epoch.setdefault(current_epoch, EpochMetrics())

            if current_epoch is None:
                continue

            train = _extract_train_metrics(line)
            if train is not None:
                train_loss, loss_rot, loss_trans = train
                em = metrics_by_epoch.setdefault(current_epoch, EpochMetrics())
                em.train_loss = train_loss
                em.loss_rot = loss_rot
                em.loss_trans = loss_trans
                tr = _extract_t_r(line)
                if tr is not None:
                    em.train_t_cm, em.train_r_deg = tr
                continue

            if "Val" in line:
                val_loss = _extract_val_loss(line)
                if val_loss is not None:
                    em = metrics_by_epoch.setdefault(current_epoch, EpochMetrics())
                    em.val_loss = val_loss
                vr = _extract_t_r(line)
                if vr is not None:
                    em = metrics_by_epoch.setdefault(current_epoch, EpochMetrics())
                    em.val_t_cm, em.val_r_deg = vr

    # Keep only epochs that have the metric(s) we need; still align by epoch.
    epochs_sorted = sorted(metrics_by_epoch.keys())

    epochs: List[int] = []
    train_losses: List[float] = []
    loss_rots: List[float] = []
    loss_transes: List[float] = []
    val_losses: List[float] = []
    train_t_cms: List[float] = []
    train_r_degs: List[float] = []
    val_t_cms: List[float] = []
    val_r_degs: List[float] = []

    for e in epochs_sorted:
        em = metrics_by_epoch[e]
        if (
            em.train_loss is None
            or em.loss_rot is None
            or em.loss_trans is None
            or em.val_loss is None
            or em.train_t_cm is None
            or em.train_r_deg is None
            or em.val_t_cm is None
            or em.val_r_deg is None
        ):
            # Skip incomplete epochs (e.g., truncated logs)
            continue
        epochs.append(e)
        train_losses.append(em.train_loss)
        loss_rots.append(em.loss_rot)
        loss_transes.append(em.loss_trans)
        val_losses.append(em.val_loss)
        train_t_cms.append(em.train_t_cm)
        train_r_degs.append(em.train_r_deg)
        val_t_cms.append(em.val_t_cm)
        val_r_degs.append(em.val_r_deg)

    return (
        epochs,
        train_losses,
        loss_rots,
        loss_transes,
        val_losses,
        train_t_cms,
        train_r_degs,
        val_t_cms,
        val_r_degs,
    )


def plot_and_save(
    epochs: List[int],
    train_losses: List[float],
    loss_rots: List[float],
    loss_transes: List[float],
    val_losses: List[float],
    train_t_cms: List[float],
    train_r_degs: List[float],
    val_t_cms: List[float],
    val_r_degs: List[float],
    out_dir: str,
    prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1) train loss
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, train_losses, label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}train_loss.png"))
    plt.close()

    # 2) loss_rot + loss_trans
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, loss_rots, label="loss_rot")
    plt.plot(epochs, loss_transes, label="loss_trans")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Rotation / Translation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}rot_trans_loss.png"))
    plt.close()

    # 3) val loss
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}val_loss.png"))
    plt.close()

    # 4) train T + R
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, train_t_cms, label="train T (cm)")
    plt.plot(epochs, train_r_degs, label="train R (deg)")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Train T / R")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}train_TR.png"))
    plt.close()

    # 5) val T + R
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, val_t_cms, label="val T (cm)")
    plt.plot(epochs, val_r_degs, label="val R (deg)")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Val T / R")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}val_TR.png"))
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse training log and plot train/val losses."
    )
    parser.add_argument(
        "--log",
        type=str,
        default=os.path.join("checkpoints_2", "checkpoints_2", "log.txt"),
        help="Path to log.txt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for plots (default: <log_dir>/plots)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Filename prefix for output images",
    )

    args = parser.parse_args()
    log_path = args.log

    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"log file not found: {log_path}")

    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(log_path)), "plots")

    (
        epochs,
        train_losses,
        loss_rots,
        loss_transes,
        val_losses,
        train_t_cms,
        train_r_degs,
        val_t_cms,
        val_r_degs,
    ) = parse_log(log_path)

    if not epochs:
        raise RuntimeError(
            "No complete epochs parsed. Check log format or whether log is truncated."
        )

    plot_and_save(
        epochs,
        train_losses,
        loss_rots,
        loss_transes,
        val_losses,
        train_t_cms,
        train_r_degs,
        val_t_cms,
        val_r_degs,
        out_dir,
        args.prefix,
    )

    print(f"Parsed epochs: {len(epochs)} (from {epochs[0]} to {epochs[-1]})")
    print(f"Saved plots to: {out_dir}")
    print(f"- {os.path.join(out_dir, args.prefix + 'train_loss.png')}")
    print(f"- {os.path.join(out_dir, args.prefix + 'rot_trans_loss.png')}")
    print(f"- {os.path.join(out_dir, args.prefix + 'val_loss.png')}")
    print(f"- {os.path.join(out_dir, args.prefix + 'train_TR.png')}")
    print(f"- {os.path.join(out_dir, args.prefix + 'val_TR.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
