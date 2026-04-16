# kalman_nonlinear.py
#
# Same Kalman filter approach as the linear version
# but operating in the UNet autoencoder latent space
# instead of the TSVD basis.
#
# The AE latent space captures nonlinear structure that
# TSVD misses - things like the curved edges of the fire
# plume and intensity variations within the hotspot.
# This generally gives better assimilation MSE.
#
# In my experiments this cut MSE from 0.081 to 0.0054
# vs the background - over 15x improvement.
#
# The Kalman update math is identical to the linear case.
# The only difference is how we encode and decode.

import numpy as np
import numpy.linalg as la
import torch

from src.compression.autoencoder import UNetAE
from src.utils.logging_config import get_logger
from src.utils.metrics import Timer

logger = get_logger(__name__)


class NonlinearAssimilator:

    def __init__(
        self,
        weights_path: str,
        mean_map: np.ndarray,
        std_map: np.ndarray,
        latent_dim: int,
        beta: float = 0.68,
    ):
        self.mean_map = mean_map
        self.std_map = std_map
        self.latent_dim = latent_dim
        self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the trained autoencoder
        H, W = mean_map.shape
        self.model = UNetAE(latent_dim, H, W).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        logger.info(
            f"loaded UNetAE weights from {weights_path} | " f"device: {self.device}"
        )

    def run(self, background_path: str, obs_path: str, truth_path: str) -> dict:

        X_b = np.load(background_path, mmap_mode="r")
        X_obs = np.load(obs_path, mmap_mode="r")
        X_truth = np.load(truth_path, mmap_mode="r")

        N, H, W = X_obs.shape

        logger.info(
            f"nonlinear assimilation | "
            f"{N} frames | spatial: {H}x{W} | "
            f"latent dim: {self.latent_dim}"
        )

        # encode all three datasets into AE latent space
        logger.info("encoding to AE latent space...")
        with Timer("encoding") as t_enc:
            Zb = self._encode_all(X_b, N)
            Zo = self._encode_all(X_obs, N)
            Zt = self._encode_all(X_truth, N)

        logger.info(f"encoding complete in {t_enc.elapsed}s")

        # background MSE before assimilation
        bg_flat = X_b.reshape(N, -1).astype(float)
        truth_flat = X_truth[:N].reshape(N, -1).astype(float)
        mse_background = float(np.mean((bg_flat - truth_flat) ** 2))
        logger.info(f"background MSE: {mse_background:.3e}")

        # build Kalman gain in AE latent space
        logger.info("building Kalman gain in AE latent space...")
        K_gain = self._build_kalman_gain(Zb)

        # latent space Kalman update
        with Timer("latent Kalman update") as t_latent:
            Za = Zb + (Zo - Zb).dot(K_gain.T)

        mse_latent = float(np.mean((Za - Zt) ** 2))
        logger.info(f"latent MSE: {mse_latent:.3e}")

        # decode back to physical space
        with Timer("decoding") as t_decode:
            X_a = self._decode_all(Za, H, W)

        mse_physical = float(np.mean((X_a.reshape(N, -1) - truth_flat) ** 2))

        logger.info(f"analysis MSE: {mse_physical:.3e}")
        logger.info(
            f"improvement over background: "
            f"{((mse_background - mse_physical) / mse_background * 100):.1f}%"
        )

        return {
            "mse_background": mse_background,
            "mse_latent": mse_latent,
            "mse_physical": mse_physical,
            "encoding_time_s": t_enc.elapsed,
            "latent_update_time_s": t_latent.elapsed,
            "decoding_time_s": t_decode.elapsed,
            "beta": self.beta,
            "n_frames": N,
        }

    def _encode_all(self, X: np.ndarray, N: int) -> np.ndarray:
        Z = np.zeros((N, self.latent_dim))

        with torch.no_grad():
            for i in range(N):
                img = (X[i].astype(float) - self.mean_map) / self.std_map

                t = torch.from_numpy(img)[None, None].float().to(self.device)

                z, _, _, _ = self.model.encode(t)
                Z[i] = z.cpu().numpy()

        return Z

    def _decode_all(self, Z: np.ndarray, H: int, W: int) -> np.ndarray:
        X_rec = np.zeros((len(Z), H, W))

        with torch.no_grad():
            # batch decode for speed
            Z_tensor = torch.from_numpy(Z.astype(np.float32)).to(self.device)

            rec = self.model.decode(Z_tensor)
            X_rec = rec.cpu().numpy()[:, 0, :, :]

        # un-normalise back to physical space
        return X_rec * self.std_map + self.mean_map

    def _build_kalman_gain(self, Zb: np.ndarray) -> np.ndarray:
        Zb_ctr = Zb - Zb.mean(axis=0)
        B = np.cov(Zb_ctr, rowvar=False, ddof=1)

        # same R = beta * I as linear case
        # reusing beta=0.68 since it was tuned on
        # the same dataset - consistent comparison
        R = self.beta * np.eye(self.latent_dim)
        K_gain = B.dot(la.inv(B + R))

        logger.info(
            f"Kalman gain built | " f"beta={self.beta} | " f"B shape: {B.shape}"
        )

        return K_gain
