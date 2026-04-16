# test_compression.py
#
# Tests for the compression modules.
# I'm testing behaviour not implementation -
# if I refactor internals these tests should
# still pass as long as the outputs are correct.

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD
from src.compression.tsvd import TSVDCompressor

# ── helpers ──────────────────────────────────────────────────


def make_fake_data(
    n_samples: int = 100, h: int = 32, w: int = 32, seed: int = 0
) -> np.ndarray:
    # small synthetic fire-like data
    # sparse binary array - mimics the real dataset
    rng = np.random.default_rng(seed)
    data = np.zeros((n_samples, h, w), dtype=np.float32)

    # put a small "hotspot" in a random location per frame
    for i in range(n_samples):
        cx = rng.integers(8, 24)
        cy = rng.integers(8, 24)
        data[i, cx - 2 : cx + 2, cy - 2 : cy + 2] = 1.0

    return data


def save_fake_data(path: str, n: int = 100) -> np.ndarray:
    data = make_fake_data(n_samples=n)
    np.save(path, data)
    return data


# ── TSVDCompressor tests ──────────────────────────────────────


class TestTSVDCompressor:

    def test_fit_returns_self(self, tmp_path):
        # fit() should return the compressor instance
        # so we can chain calls if needed
        train_path = str(tmp_path / "train.npy")
        save_fake_data(train_path)

        compressor = TSVDCompressor(n_components=10, batch_size=20)
        result = compressor.fit(train_path)

        assert result is compressor

    def test_mean_shape_after_fit(self, tmp_path):
        # mean vector should have same length as flattened frame
        train_path = str(tmp_path / "train.npy")
        save_fake_data(train_path)

        compressor = TSVDCompressor(n_components=10, batch_size=20)
        compressor.fit(train_path)

        assert compressor.mean.shape == (32 * 32,)

    def test_model_fitted_after_fit(self, tmp_path):
        # model should not be None after fitting
        train_path = str(tmp_path / "train.npy")
        save_fake_data(train_path)

        compressor = TSVDCompressor(n_components=10, batch_size=20)
        compressor.fit(train_path)

        assert compressor.model is not None

    def test_transform_shape(self, tmp_path):
        # transform should reduce spatial dims to n_components
        train_path = str(tmp_path / "train.npy")
        data = save_fake_data(train_path)

        compressor = TSVDCompressor(n_components=10, batch_size=20)
        compressor.fit(train_path)

        flat = data[:5].reshape(5, -1).astype(float)
        Z = compressor.transform(flat)

        assert Z.shape == (5, 10)

    def test_inverse_transform_shape(self, tmp_path):
        # inverse transform should recover original spatial dims
        train_path = str(tmp_path / "train.npy")
        data = save_fake_data(train_path)

        compressor = TSVDCompressor(n_components=10, batch_size=20)
        compressor.fit(train_path)

        flat = data[:5].reshape(5, -1).astype(float)
        Z = compressor.transform(flat)
        rec = compressor.inverse_transform(Z)

        assert rec.shape == flat.shape

    def test_reconstruction_mse_reasonable(self, tmp_path):
        # with enough components MSE should be low
        # using 20 components on 32x32 synthetic data
        # so this should be well under 0.1
        train_path = str(tmp_path / "train.npy")
        test_path = str(tmp_path / "test.npy")

        save_fake_data(train_path, n=200)
        save_fake_data(test_path, n=50)

        compressor = TSVDCompressor(n_components=20, batch_size=50)
        compressor.fit(train_path)
        mse, _ = compressor.evaluate(test_path)

        assert mse < 0.1, f"MSE too high: {mse:.4f}"

    def test_evaluate_raises_before_fit(self, tmp_path):
        # should raise clearly if evaluate called before fit
        test_path = str(tmp_path / "test.npy")
        save_fake_data(test_path)

        compressor = TSVDCompressor(n_components=10)

        with pytest.raises(RuntimeError, match="fit()"):
            compressor.evaluate(test_path)

    def test_n_components_respected(self, tmp_path):
        # model should have exactly n_components components
        train_path = str(tmp_path / "train.npy")
        save_fake_data(train_path)

        n = 8
        compressor = TSVDCompressor(n_components=n, batch_size=20)
        compressor.fit(train_path)

        assert compressor.model.n_components == n
