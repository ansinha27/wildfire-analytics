# test_assimilation.py
#
# Tests for the Kalman filter assimilation.
# I'm using small synthetic data so tests run fast
# without needing the real 4GB dataset.

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD
from src.assimilation.kalman_linear import LinearAssimilator

# ── helpers ───────────────────────


def make_fitted_compressor(
    n_samples: int = 200, h: int = 32, w: int = 32, n_components: int = 10
):
    # builds and fits a real TruncatedSVD on synthetic data
    # so assimilation tests have a proper compressor to use
    rng = np.random.default_rng(42)
    data = rng.random((n_samples, h, w)).astype(np.float32)
    flat = data.reshape(n_samples, -1).astype(float)

    mean = flat.mean(axis=0)
    centered = flat - mean

    model = TruncatedSVD(n_components=n_components, random_state=0)
    model.fit(centered)

    return model, mean


def save_fake_npy(path: str, shape: tuple, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.random(shape).astype(np.float32)
    np.save(path, data)
    return data


# ── LinearAssimilator tests ───────────────────────────────────


class TestLinearAssimilator:

    def test_init(self):
        # should initialise without errors
        compressor, mean = make_fitted_compressor()
        assimilator = LinearAssimilator(compressor=compressor, mean=mean, beta=0.68)
        assert assimilator.beta == 0.68
        assert assimilator.K is None

    def test_kalman_gain_shape(self):
        # Kalman gain should be square (K_dim x K_dim)
        compressor, mean = make_fitted_compressor(n_components=10)
        assimilator = LinearAssimilator(compressor=compressor, mean=mean)

        # build some fake latent vectors
        rng = np.random.default_rng(0)
        Zb = rng.random((50, 10))

        K_gain = assimilator._build_kalman_gain(Zb, 10)

        assert K_gain.shape == (10, 10)

    def test_kalman_gain_eigenvalues_bounded(self):
        # eigenvalues of Kalman gain should be in (0, 1)
        # because K = B(B+R)^-1 and B,R are positive definite
        compressor, mean = make_fitted_compressor(n_components=10)
        assimilator = LinearAssimilator(compressor=compressor, mean=mean, beta=0.68)

        rng = np.random.default_rng(0)
        Zb = rng.random((50, 10))
        K_gain = assimilator._build_kalman_gain(Zb, 10)

        eigvals = np.real(np.linalg.eigvals(K_gain))

        assert np.all(eigvals > 0), "eigenvalues should be positive"
        assert np.all(eigvals < 1), "eigenvalues should be < 1"

    def test_assimilation_reduces_mse(self, tmp_path):
        # the Kalman update should reduce latent-space MSE
        # vs the background - we test in latent space here
        # because the synthetic compressor isn't trained on
        # the same distribution as the test data, so
        # physical-space MSE isn't a fair measure on toy data.
        # on real fire data the physical MSE improves too
        # (verified in the full pipeline runs)
        compressor, mean = make_fitted_compressor(n_samples=300, n_components=10)

        shape = (20, 32, 32)
        n_feats = 32 * 32
        K_dim = 10

        # truth
        truth = np.random.default_rng(0).random(shape).astype(np.float32)

        # background - forecast with some error
        background = truth + np.random.default_rng(1).normal(0, 0.1, shape).astype(
            np.float32
        )

        # observations - closer to truth than background
        obs = truth + np.random.default_rng(2).normal(0, 0.05, shape).astype(np.float32)

        bg_path = str(tmp_path / "bg.npy")
        obs_path = str(tmp_path / "obs.npy")
        truth_path = str(tmp_path / "truth.npy")

        np.save(bg_path, background)
        np.save(obs_path, obs)
        np.save(truth_path, truth)

        assimilator = LinearAssimilator(compressor=compressor, mean=mean, beta=0.68)

        results = assimilator.run(
            background_path=bg_path, obs_path=obs_path, truth_path=truth_path
        )

        # latent MSE should be lower than background latent MSE
        # this is what the Kalman filter directly optimises
        # project background and truth manually to compare
        Zb = np.zeros((20, K_dim))
        Zt = np.zeros((20, K_dim))

        for i in range(20):
            flat_b = background[i].reshape(-1).astype(float) - mean
            flat_t = truth[i].reshape(-1).astype(float) - mean
            Zb[i] = compressor.transform(flat_b[None, :])[0]
            Zt[i] = compressor.transform(flat_t[None, :])[0]

        mse_bg_latent = float(np.mean((Zb - Zt) ** 2))

        assert (
            results["mse_latent"] < mse_bg_latent
        ), "Kalman update should reduce latent space MSE"

    def test_run_returns_expected_keys(self, tmp_path):
        # results dict should always have these keys
        # so downstream code can rely on them
        compressor, mean = make_fitted_compressor(n_components=10)

        shape = (20, 32, 32)
        for name, seed in [("bg", 0), ("obs", 1), ("truth", 2)]:
            np.save(
                str(tmp_path / f"{name}.npy"),
                np.random.default_rng(seed).random(shape).astype(np.float32),
            )

        assimilator = LinearAssimilator(compressor=compressor, mean=mean)

        results = assimilator.run(
            background_path=str(tmp_path / "bg.npy"),
            obs_path=str(tmp_path / "obs.npy"),
            truth_path=str(tmp_path / "truth.npy"),
        )

        expected_keys = [
            "mse_background",
            "mse_latent",
            "mse_physical",
            "latent_update_time_s",
            "reconstruction_time_s",
            "beta",
            "n_frames",
        ]

        for key in expected_keys:
            assert key in results, f"missing key: {key}"

    def test_beta_affects_gain(self):
        # higher beta means less trust in observations
        # so the Kalman gain should be smaller
        compressor, mean = make_fitted_compressor(n_components=10)

        rng = np.random.default_rng(0)
        Zb = rng.random((50, 10))

        low_beta = LinearAssimilator(compressor=compressor, mean=mean, beta=0.1)
        high_beta = LinearAssimilator(compressor=compressor, mean=mean, beta=10.0)

        K_low = low_beta._build_kalman_gain(Zb, 10)
        K_high = high_beta._build_kalman_gain(Zb, 10)

        # frobenius norm of gain should be larger for low beta
        assert np.linalg.norm(K_low) > np.linalg.norm(
            K_high
        ), "lower beta should give larger Kalman gain"
