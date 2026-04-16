import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# I save all figures to outputs/ so I can review them
# without the script blocking on plt.show()
OUTPUT_DIR = Path("outputs/figures")


def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Figure saved: {path}")


def plot_frames(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    n: int = 5,
    title: str = "reconstruction",
    indices: list = None
) -> None:
    # side by side comparison of original vs reconstructed frames
    # useful sanity check after compression or assimilation
    if indices is None:
        indices = np.random.choice(len(originals), size=n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))

    for row, idx in enumerate(indices):
        orig = originals[idx]
        rec = reconstructions[idx]

        vmin = min(orig.min(), rec.min())
        vmax = max(orig.max(), rec.max())

        axes[row, 0].imshow(orig, cmap="hot", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Original [{idx}]")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rec, cmap="hot", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Reconstructed [{idx}]")
        axes[row, 1].axis("off")

    plt.tight_layout()
    _save(fig, f"{title}_comparison.png")


def plot_cumulative_variance(
    cumvar: np.ndarray,
    k: int,
    threshold: float = 0.95
) -> None:
    # helps justify the choice of n_components
    # I used this to pick K=114 in the notebook
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, ".-", color="steelblue")
    ax.axvline(k, color="green", linestyle="--", label=f"K={k}")
    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"{int(threshold * 100)}% variance"
    )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Variance explained by TSVD components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "cumulative_variance.png")


def plot_training_curve(
    train_losses: list,
    val_losses: list
) -> None:
    # sanity check that the autoencoder actually converged
    # and didn't overfit
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(train_losses, label="train loss", color="steelblue")
    ax.plot(val_losses, label="val loss", color="orange")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Autoencoder training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "ae_training_curve.png")


def plot_assimilation_results(
    truth: np.ndarray,
    background: np.ndarray,
    analysis: np.ndarray,
    idx: int = 0,
    label: str = "assimilation"
) -> None:
    # three-way comparison: what the model predicted,
    # what we observed, and what the analysis produced
    # this is the key result for tasks 4 and 5
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    vmin = truth.min()
    vmax = truth.max()

    for ax, img, title in zip(
        axes,
        [truth, background, analysis],
        ["Truth", "Background", "Analysis"]
    ):
        im = ax.imshow(img, cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    plt.tight_layout()
    _save(fig, f"{label}_result.png")


def plot_fusion_result(
    obs1: np.ndarray,
    obs2: np.ndarray,
    fused: np.ndarray,
    binary: np.ndarray
) -> None:
    # shows the two input observations and the fused output
    # side by side - makes it easy to explain in an interview
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, img, title in zip(
        axes,
        [obs1, obs2, fused, binary],
        ["Observation 1", "Observation 2", "Fused", "Thresholded"]
    ):
        ax.imshow(img, cmap="hot", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    _save(fig, "fusion_result.png")