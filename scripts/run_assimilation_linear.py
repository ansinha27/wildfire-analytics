# run_assimilation_linear.py
#
# Kalman filter data assimilation in TSVD latent space.
# Requires compression stage to have run first.
#
# Usage:
#   python scripts/run_assimilation_linear.py
#   python scripts/run_assimilation_linear.py --beta 0.5

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    get_storage,
    data_config,
    assimilation_config,
    path_config
)
from src.assimilation.kalman_linear import LinearAssimilator
from src.utils.logging_config import get_logger
from src.utils.visualisation import plot_assimilation_results
import numpy as np

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kalman filter assimilation in TSVD space"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=assimilation_config.beta,
        help=f"observation error scale (default: {assimilation_config.beta})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = get_storage()

    logger.info("=" * 50)
    logger.info("DATA ASSIMILATION — Linear (TSVD)")
    logger.info("=" * 50)
    logger.info(f"beta       : {args.beta}")
    logger.info(f"background : {data_config.background_path}")
    logger.info(f"obs        : {data_config.obs_path}")

    # load compression artifacts
    compressor = storage.load_model(path_config.tsvd_model)
    mean = storage.load_array(path_config.mean_train)

    assimilator = LinearAssimilator(
        compressor=compressor,
        mean=mean,
        beta=args.beta
    )

    results = assimilator.run(
        background_path=data_config.background_path,
        obs_path=data_config.obs_path,
        truth_path=data_config.test_path
    )

    # visualise a few frames
    X_b = np.load(data_config.background_path, mmap_mode="r")
    X_truth = np.load(data_config.test_path, mmap_mode="r")

    storage.save_results(results, "assimilation_linear_results")

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"background MSE : {results['mse_background']:.3e}")
    logger.info(f"analysis MSE   : {results['mse_physical']:.3e}")
    logger.info(
        f"improvement    : "
        f"{((results['mse_background'] - results['mse_physical']) / results['mse_background'] * 100):.1f}%"
    )
    logger.info(
        f"latent update  : {results['latent_update_time_s']}s"
    )


if __name__ == "__main__":
    main()