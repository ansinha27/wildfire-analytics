# kalman_linear.py
#
# Data assimilation using a Kalman filter in the TSVD
# latent space.
#
# Rather than assimilating in full 256x256 pixel space
# (65536 dimensions), I project everything into the
# 114-dimensional TSVD basis first, run the Kalman update
# there, then reconstruct back to physical space.
#
# This is orders of magnitude cheaper and works well
# because the TSVD basis filters out noise that isn't
# captured by the leading modes anyway.
#
# Covariance choices:
#   B = empirical covariance of background ensemble
#       in latent space. Full matrix so every mode
#       can borrow strength from correlated modes.
#   R = beta * I  (diagonal, scaled by beta)
#       I don't know the true sensor noise so I use
#       a simple isotropic form and tune beta by
#       sweeping validation MSE. beta=0.68 was optimal.
#
# The Kalman update:
#   Z_a = Z_b + (Z_o - Z_b) K^T
#   where K = B (B + R)^-1

import numpy as np
import numpy.linalg as la
from typing import TYPE_CHECKING

# only import for type hints
if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD

from src.utils.logging_config import get_logger
from src.utils.metrics import Timer

logger = get_logger(__name__)


class LinearAssimilator:

    def __init__(
        self,
        compressor,       # fitted TruncatedSVD model
        mean: np.ndarray,
        beta: float = 0.68
    ):
        self.compressor = compressor
        self.mean = mean
        self.beta = beta
        self.K = None     # Kalman gain — built during run()

    def run(
        self,
        background_path: str,
        obs_path: str,
        truth_path: str
    ) -> dict:

        X_b     = np.load(background_path, mmap_mode="r")
        X_obs   = np.load(obs_path,        mmap_mode="r")
        X_truth = np.load(truth_path,      mmap_mode="r")

        N, H, W = X_b.shape
        n_feats = H * W
        K_dim   = self.compressor.n_components

        logger.info(
            f"assimilating {N} frames | "
            f"spatial: {H}x{W} | "
            f"latent dim: {K_dim}"
        )

        # project all three datasets into latent space
        logger.info("projecting to latent space...")
        Zb, Zo, Zt = self._project_all(
            X_b, X_obs, X_truth, N, n_feats
        )

        # background MSE before assimilation
        bg_flat    = X_b.reshape(N, -1).astype(float)
        truth_flat = X_truth[:N].reshape(N, -1).astype(float)
        mse_background = float(
            np.mean((bg_flat - truth_flat) ** 2)
        )
        logger.info(f"background MSE: {mse_background:.3e}")

        # build Kalman gain matrix
        logger.info("building Kalman gain matrix...")
        self.K = self._build_kalman_gain(Zb, K_dim)

        # latent space Kalman update
        with Timer("latent Kalman update") as t_latent:
            Za = Zb + (Zo - Zb).dot(self.K.T)

        mse_latent = float(np.mean((Za - Zt) ** 2))
        logger.info(f"latent MSE: {mse_latent:.3e}")
        logger.info(f"latent update time: {t_latent.elapsed}s")

        # reconstruct to physical space
        with Timer("reconstruction") as t_recon:
            X_a_flat = (
                self.compressor.inverse_transform(Za) + self.mean
            )

        mse_physical = float(
            np.mean((X_a_flat - truth_flat) ** 2)
        )

        improvement = (
            (mse_background - mse_physical)
            / mse_background * 100
        )

        logger.info(f"analysis MSE: {mse_physical:.3e}")
        logger.info(
            f"improvement over background: {improvement:.1f}%"
        )

        return {
            "mse_background":        mse_background,
            "mse_latent":            mse_latent,
            "mse_physical":          mse_physical,
            "latent_update_time_s":  t_latent.elapsed,
            "reconstruction_time_s": t_recon.elapsed,
            "beta":                  self.beta,
            "n_frames":              N
        }

    def _project_all(
        self,
        X_b:     np.ndarray,
        X_obs:   np.ndarray,
        X_truth: np.ndarray,
        N:       int,
        n_feats: int
    ) -> tuple:
        K_dim = self.compressor.n_components
        Zb = np.zeros((N, K_dim))
        Zo = np.zeros((N, K_dim))
        Zt = np.zeros((N, K_dim))

        for i in range(N):
            Zb[i] = self._project(X_b[i],     n_feats)
            Zo[i] = self._project(X_obs[i],   n_feats)
            Zt[i] = self._project(X_truth[i], n_feats)

        return Zb, Zo, Zt

    def _project(
        self,
        frame:   np.ndarray,
        n_feats: int
    ) -> np.ndarray:
        # flatten, centre, project into latent space
        flat = frame.reshape(-1).astype(float) - self.mean
        return self.compressor.transform(flat[None, :])[0]

    def _build_kalman_gain(
        self,
        Zb:    np.ndarray,
        K_dim: int
    ) -> np.ndarray:
        # empirical background covariance in latent space
        # full matrix so correlated modes share information
        Zb_ctr = Zb - Zb.mean(axis=0)
        B = np.cov(Zb_ctr, rowvar=False, ddof=1)

        # R = beta * I
        # diagonal observation error covariance
        # beta tuned by validation sweep - 0.68 was optimal
        R = self.beta * np.eye(K_dim)

        # Kalman gain: K = B (B + R)^-1
        K_gain = B.dot(la.inv(B + R))

        logger.info(
            f"Kalman gain built | "
            f"beta={self.beta} | "
            f"B shape: {B.shape}"
        )

        return K_gain