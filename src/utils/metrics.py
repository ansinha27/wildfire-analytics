import numpy as np
import time
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # straightforward MSE - just making sure shapes match first
    # caught a few silent bugs this way during development
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} "
            f"vs reconstructed {reconstructed.shape}"
        )
    mse = float(np.mean((original - reconstructed) ** 2))
    return mse


def compute_compression_ratio(
    original_shape: tuple,
    compressed_dim: int
) -> float:
    # how much smaller is the compressed representation?
    # e.g. 256x256 -> 114 coefficients
    original_size = int(np.prod(original_shape))
    ratio = (1 - compressed_dim / original_size) * 100
    return round(ratio, 2)


def evaluate_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    label: str = ""
) -> dict:
    # runs all the metrics I care about in one go
    # returns a dict so I can log it or save it to cosmos later
    mse = compute_mse(original, reconstructed)

    n_samples, *spatial = original.shape
    compression_ratio = compute_compression_ratio(
        tuple(spatial), 1
    )

    results = {
        "label": label,
        "mse": mse,
        "n_samples": n_samples,
        "spatial_shape": tuple(spatial),
    }

    logger.info(f"[{label}] MSE: {mse:.3e}")
    logger.info(f"[{label}] Samples evaluated: {n_samples}")

    return results


class Timer:
    # simple context manager so I can time any block cleanly
    # usage:
    #   with Timer("SVD fit") as t:
    #       model.fit(data)
    #   print(t.elapsed)

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.time() - self._start, 2)
        logger.info(f"[{self.label}] completed in {self.elapsed}s")