# tsvd.py
#
# Linear compression using Truncated SVD.
# I memory-map the training data so I never load the full
# array into RAM - this was essential for the 4GB+ dataset.
#
# The key insight from EDA was that >96% of pixels are zero
# (sparse fire masks) so a linear basis works really well here.
# 114 components captures 95% of the variance.

import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.metrics import Timer, compute_mse

logger = get_logger(__name__)


class TSVDCompressor:
    # fits a TruncatedSVD basis on large fire mask datasets
    # and reconstructs them from a compressed representation
    #
    # why TruncatedSVD over PCA?
    # - works directly on memmap arrays
    # - single pass, no convergence issues
    # - much faster than IncrementalPCA on this data
    #   (28s vs 135s in my experiments)

    def __init__(self, n_components: int = 114, batch_size: int = 50, n_iter: int = 7):
        self.n_components = n_components
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.model = None
        self.mean = None
        self._spatial = None

    def fit(self, train_path: str) -> "TSVDCompressor":
        logger.info(f"loading training data from {train_path}")

        X = np.load(train_path, mmap_mode="r")
        n_train, *spatial = X.shape
        self._spatial = tuple(spatial)
        n_feats = int(np.prod(spatial))

        logger.info(
            f"training data: {n_train} frames, " f"spatial shape {self._spatial}"
        )

        # compute mean in batches - never load full array
        # into RAM at once
        logger.info("computing global mean in batches...")
        self.mean = self._compute_mean(X, n_train, n_feats)

        # write centred data to a memmap on disk
        # so SVD doesn't need to re-centre on every pass
        logger.info("writing centred memmap to disk...")
        centered = self._build_centered_memmap(X, n_train, n_feats)

        # fit the SVD
        logger.info(
            f"fitting TruncatedSVD with " f"n_components={self.n_components}..."
        )
        self.model = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            n_iter=self.n_iter,
            random_state=0,
        )

        with Timer("TruncatedSVD fit") as t:
            self.model.fit(centered)

        explained = self.model.explained_variance_ratio_.sum()
        logger.info(
            f"SVD fit complete | "
            f"variance explained: {explained:.3f} | "
            f"time: {t.elapsed}s"
        )

        return self

    def transform(self, X_flat: np.ndarray) -> np.ndarray:
        # project centred data into the latent space
        centred = X_flat - self.mean
        return self.model.transform(centred)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        # reconstruct from latent coefficients
        return self.model.inverse_transform(Z) + self.mean

    def evaluate(self, test_path: str) -> tuple[float, float]:
        # reconstruct test frames and measure MSE
        # returns (mse, reconstruction_time_seconds)
        if self.model is None:
            raise RuntimeError("call fit() before evaluate()")

        X_test = np.load(test_path, mmap_mode="r")
        n_test, *spatial = X_test.shape
        n_feats = int(np.prod(spatial))

        sse = 0.0

        with Timer("reconstruction") as t:
            for i in range(0, n_test, self.batch_size):
                batch = X_test[i : i + self.batch_size]
                batch_flat = batch.reshape(-1, n_feats).astype(float)

                Z = self.transform(batch_flat)
                rec = self.inverse_transform(Z)

                sse += ((batch_flat - rec) ** 2).sum()

        mse = sse / (n_test * n_feats)

        logger.info(f"test MSE: {mse:.3e}")
        logger.info(f"reconstruction time: {t.elapsed}s")

        return mse, t.elapsed

    def _compute_mean(self, X: np.ndarray, n_train: int, n_feats: int) -> np.ndarray:
        mean = np.zeros(n_feats, dtype=float)
        total = 0

        for i in range(0, n_train, self.batch_size):
            block = X[i : i + self.batch_size]
            block_flat = block.reshape(-1, n_feats).astype(float)
            mean += block_flat.sum(axis=0)
            total += block_flat.shape[0]

        return mean / total

    def _build_centered_memmap(
        self, X: np.ndarray, n_train: int, n_feats: int
    ) -> np.ndarray:
        # write centred data to disk as float32 memmap
        # float32 halves memory vs float64 with negligible
        # impact on SVD accuracy for this data
        path = Path("models/train_cent.dat")
        path.parent.mkdir(parents=True, exist_ok=True)

        centered = np.memmap(
            str(path), mode="w+", dtype="float32", shape=(n_train, n_feats)
        )

        for i in range(0, n_train, self.batch_size):
            block = X[i : i + self.batch_size]
            block_flat = block.reshape(-1, n_feats).astype(float)
            centered[i : i + block_flat.shape[0]] = (block_flat - self.mean).astype(
                "float32"
            )

        centered.flush()
        return centered
