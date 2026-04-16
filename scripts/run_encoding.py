# run_encoding.py
#
# Trains the UNet autoencoder and evaluates
# reconstruction MSE on test data.
#
# Usage:
#   python scripts/run_encoding.py
#   python scripts/run_encoding.py --n_epochs 50

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_storage, data_config, ae_config, path_config
from src.compression.autoencoder import AutoencoderCompressor
from src.utils.logging_config import get_logger
from src.utils.metrics import Timer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nonlinear compression using UNet Autoencoder"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=ae_config.n_epochs,
        help=f"max training epochs (default: {ae_config.n_epochs})"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=ae_config.latent_dim,
        help="bottleneck size"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=ae_config.batch_size,
        help="training batch size"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = get_storage()

    logger.info("=" * 50)
    logger.info("NONLINEAR COMPRESSION — UNet Autoencoder")
    logger.info("=" * 50)
    logger.info(f"latent_dim : {args.latent_dim}")
    logger.info(f"n_epochs   : {args.n_epochs}")
    logger.info(f"batch_size : {args.batch_size}")
    logger.info(f"train data : {data_config.train_path}")

    encoder = AutoencoderCompressor(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=ae_config.lr,
        alpha_mse=ae_config.alpha_mse,
        alpha_edge=ae_config.alpha_edge,
        patience=ae_config.patience
    )

    with Timer("full encoding pipeline") as t:
        encoder.fit(data_config.train_path)

    logger.info("model artifacts saved")

    # evaluate
    mse, recon_time = encoder.evaluate(data_config.test_path)

    results = {
        "stage": "encoding",
        "latent_dim": args.latent_dim,
        "training_time_s": t.elapsed,
        "reconstruction_time_s": recon_time,
        "mse": mse
    }
    storage.save_results(results, "encoding_results")

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"training time     : {t.elapsed}s")
    logger.info(f"reconstruction MSE: {mse:.3e}")
    logger.info(f"reconstruction time: {recon_time}s")


if __name__ == "__main__":
    main()