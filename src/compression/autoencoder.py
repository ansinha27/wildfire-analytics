# autoencoder.py
#
# Nonlinear compression using a UNet-style convolutional
# autoencoder with skip connections.
#
# I went with UNet over a plain deep AE because the fire masks
# have very sharp edges - skip connections let high-frequency
# detail bypass the bottleneck so we don't lose the hotspot
# boundaries in reconstruction.
#
# I also added a Sobel edge loss term on top of MSE to
# explicitly penalise blurry edges. That dropped the MSE
# significantly compared to MSE-only training.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

from src.utils.logging_config import get_logger
from src.utils.metrics import Timer

logger = get_logger(__name__)


class FireDataset(Dataset):
    # wraps the memmap array and applies normalisation on the fly
    # mean/std are per-pixel maps computed from training data

    def __init__(
        self,
        path: str,
        mean: np.ndarray,
        std: np.ndarray
    ):
        self.data = np.load(path, mmap_mode="r")
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = (self.data[idx].astype(float) - self.mean) / self.std
        return torch.from_numpy(img).unsqueeze(0).float()


class UNetAE(nn.Module):
    # encoder compresses down to latent_dim
    # decoder reconstructs with skip connections from encoder
    # skip connections are what make this UNet style -
    # they pass spatial detail directly from encoder to decoder

    def __init__(self, latent_dim: int, h: int, w: int):
        super().__init__()

        self.latent_dim = latent_dim
        self._feat_h = h // 8
        self._feat_w = w // 8

        # encoder - progressively halve spatial dims
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(True)
        )

        # bottleneck - this is the latent space
        self.fc_enc = nn.Linear(
            256 * self._feat_h * self._feat_w, latent_dim
        )
        self.fc_dec = nn.Linear(
            latent_dim, 256 * self._feat_h * self._feat_w
        )

        # decoder - progressively double spatial dims
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(True)
        )
        self.dec1 = nn.Conv2d(32, 1, 3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return self.fc_enc(e4.view(e4.size(0), -1)), e1, e2, e3

    def decode(
        self,
        z: torch.Tensor,
        e1=None,
        e2=None,
        e3=None
    ) -> torch.Tensor:
        b = z.size(0)
        d4 = self.fc_dec(z).view(b, 256, self._feat_h, self._feat_w)

        # add skip connections if encoder features are available
        # during assimilation we decode without them
        d3 = self.dec4(d4) + (e3 if e3 is not None else 0)
        d2 = self.dec3(d3) + (e2 if e2 is not None else 0)
        d1 = self.dec2(d2) + (e1 if e1 is not None else 0)

        return self.dec1(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, e1, e2, e3 = self.encode(x)
        return self.decode(z, e1, e2, e3)


class EarlyStopping:
    # stops training when val loss stops improving
    # saves the best checkpoint so I can reload it after

    def __init__(
        self,
        patience: int = 10,
        delta: float = 1e-5,
        path: str = "models/unet_ae_checkpoint.pt"
    ):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_loss": self.best_loss
            }, self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _sobel_edge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    # penalise differences in edge structure
    # this keeps the sharp fire boundaries intact
    # without this the reconstruction tends to blur the hotspot
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0).unsqueeze(0)

    sobel_y = sobel_x.transpose(2, 3)

    gx_p = F.conv2d(pred, sobel_x, padding=1)
    gy_p = F.conv2d(pred, sobel_y, padding=1)
    gx_t = F.conv2d(target, sobel_x, padding=1)
    gy_t = F.conv2d(target, sobel_y, padding=1)

    return F.mse_loss(gx_p, gx_t) + F.mse_loss(gy_p, gy_t)


class AutoencoderCompressor:
    # wraps UNetAE with training, evaluation and
    # encode/decode convenience methods
    # designed to be a drop-in alongside TSVDCompressor

    def __init__(
        self,
        latent_dim: int = 114,
        batch_size: int = 32,
        n_epochs: int = 100,
        lr: float = 1e-3,
        alpha_mse: float = 1.0,
        alpha_edge: float = 0.1,
        patience: int = 10,
        delta: float = 1e-5
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.alpha_mse = alpha_mse
        self.alpha_edge = alpha_edge
        self.patience = patience
        self.delta = delta

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.mean_map = None
        self.std_map = None

        logger.info(f"using device: {self.device}")

    def fit(self, train_path: str) -> "AutoencoderCompressor":
        logger.info("computing per-pixel mean and std...")
        self.mean_map, self.std_map = self._compute_stats(train_path)

        X_mm = np.load(train_path, mmap_mode="r")
        H, W = X_mm.shape[1], X_mm.shape[2]

        self.model = UNetAE(self.latent_dim, H, W).to(self.device)

        train_ds = FireDataset(train_path, self.mean_map, self.std_map)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
        stopper = EarlyStopping(
            patience=self.patience,
            delta=self.delta
        )

        logger.info(
            f"training UNetAE | "
            f"latent_dim={self.latent_dim} | "
            f"epochs={self.n_epochs}"
        )

        train_losses, val_losses = [], []

        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._train_epoch(
                train_loader, optimizer
            )
            val_loss = self._val_epoch(train_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(
                f"epoch {epoch}/{self.n_epochs} | "
                f"train: {train_loss:.6f} | "
                f"val: {val_loss:.6f}"
            )

            stopper(val_loss, self.model, optimizer, epoch)
            if stopper.early_stop:
                logger.info(f"early stopping at epoch {epoch}")
                break

            scheduler.step()

        # reload best weights
        ckpt = torch.load(
            "models/unet_ae_checkpoint.pt",
            map_location=self.device
        )
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"loaded best model from epoch {ckpt['epoch']}"
        )

        # save stats for later use in assimilation
        np.save("models/mean_map.npy", self.mean_map)
        np.save("models/std_map.npy", self.std_map)
        np.save("models/unet_latent.npy", np.array([self.latent_dim]))

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        # encode a batch of frames to latent vectors
        self.model.eval()
        results = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                imgs = self._to_tensor(batch)
                z, _, _, _ = self.model.encode(imgs)
                results.append(z.cpu().numpy())

        return np.concatenate(results, axis=0)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        # decode latent vectors back to image space
        self.model.eval()
        results = []

        with torch.no_grad():
            for i in range(0, len(Z), self.batch_size):
                batch = torch.from_numpy(
                    Z[i:i + self.batch_size].astype(np.float32)
                ).to(self.device)
                rec = self.model.decode(batch)
                results.append(rec.cpu().numpy().squeeze(1))

        decoded = np.concatenate(results, axis=0)
        return decoded * self.std_map + self.mean_map

    def evaluate(self, test_path: str) -> tuple[float, float]:
        X_test = np.load(test_path, mmap_mode="r")
        n_test = X_test.shape[0]
        H, W = X_test.shape[1], X_test.shape[2]

        sse = 0.0

        with Timer("ae reconstruction") as t:
            for i in range(0, n_test, self.batch_size):
                batch = X_test[i:i + self.batch_size].astype(float)
                Z = self.encode(batch)
                rec = self.decode(Z)
                sse += ((batch - rec) ** 2).sum()

        mse = sse / (n_test * H * W)
        logger.info(f"test MSE: {mse:.3e}")

        return mse, t.elapsed

    def _compute_stats(
        self,
        path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.load(path, mmap_mode="r")
        n, H, W = X.shape
        mm_batch = 64

        sum_ = np.zeros((H, W), dtype=float)
        sumsq = np.zeros((H, W), dtype=float)
        count = 0

        for i in range(0, n, mm_batch):
            blk = X[i:i + mm_batch].astype(float)
            sum_ += blk.sum(axis=0)
            sumsq += (blk ** 2).sum(axis=0)
            count += blk.shape[0]

        mean = sum_ / count
        std = np.sqrt(sumsq / count - mean ** 2 + 1e-6)

        return mean, std

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        normed = (X.astype(float) - self.mean_map) / self.std_map
        return torch.from_numpy(normed).unsqueeze(1).float().to(
            self.device
        )

    def _train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        running = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            recon = self.model(batch)
            loss = (
                self.alpha_mse * F.mse_loss(recon, batch)
                + self.alpha_edge * _sobel_edge_loss(
                    recon, batch, self.device
                )
            )
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)

        return running / len(loader.dataset)

    def _val_epoch(self, loader) -> float:
        self.model.eval()
        running = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = (
                    self.alpha_mse * F.mse_loss(recon, batch)
                    + self.alpha_edge * _sobel_edge_loss(
                        recon, batch, self.device
                    )
                )
                running += loss.item() * batch.size(0)

        return running / len(loader.dataset)