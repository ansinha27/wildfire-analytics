# latent_fusion.py
#
# Fuses two satellite observations in the TSVD latent space.
#
# The idea is simple - instead of averaging raw pixels
# (which would just blur everything together), I project
# both frames into the compressed basis, average the
# latent coefficients, then reconstruct.
#
# This gives the MSE-optimal linear blend of the two inputs.
# I compared this against AE-based fusion and the linear
# approach was actually cleaner for binary fire masks -
# the AE version sometimes activated pixels where neither
# original had fire.

import numpy as np
from sklearn.decomposition import TruncatedSVD
from src.utils.logging_config import get_logger
from src.utils.metrics import Timer

logger = get_logger(__name__)


class LatentFusion:

    def __init__(self, compressor: TruncatedSVD, mean: np.ndarray):
        # takes the already-fitted TSVD model and mean vector
        # from the compression stage - no refitting needed
        self.compressor = compressor
        self.mean = mean
        self._spatial = None

    def fuse(
        self, obs_path: str, idx1: int = 0, idx2: int = -1, threshold: float = 0.5
    ) -> dict:
        # load observations
        X_obs = np.load(obs_path, mmap_mode="r")
        N, H, W = X_obs.shape
        self._spatial = (H, W)
        n_feats = H * W

        # handle negative indices the same way numpy does
        if idx2 < 0:
            idx2 = N + idx2

        logger.info(f"fusing frames {idx1} and {idx2} " f"from {N} observations")

        # project both frames into latent space
        with Timer("latent fusion") as t:
            z1 = self._project(X_obs[idx1], n_feats)
            z2 = self._project(X_obs[idx2], n_feats)

            # average in latent space
            # this is the key step - blending happens
            # in the compressed representation not in
            # pixel space
            z_fused = 0.5 * (z1 + z2)

            # reconstruct fused frame
            rec_flat = (
                self.compressor.inverse_transform(z_fused[None, :])[0] + self.mean
            )
            fused = rec_flat.reshape(H, W)

            # threshold to get a clean binary mask
            binary = (fused >= threshold).astype(int)

        logger.info(f"fusion complete in {t.elapsed}s")

        return {
            "fused": fused,
            "binary": binary,
            "obs1": X_obs[idx1],
            "obs2": X_obs[idx2],
            "idx1": idx1,
            "idx2": idx2,
            "fusion_time_s": t.elapsed,
        }

    def _project(self, frame: np.ndarray, n_feats: int) -> np.ndarray:
        # flatten, centre, project into latent space
        flat = frame.reshape(-1).astype(float) - self.mean
        return self.compressor.transform(flat[None, :])[0]
