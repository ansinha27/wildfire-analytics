# run_assimilation_nonlinear.py
#
# Kalman filter data assimilation in AE latent space.
# Requires encoding stage to have run first.
#
# Usage:
#   python scripts/run_assimilation_nonlinear.py
#   python scripts/run_assimilation_nonlinear.py --beta 0.5

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_storage, data_config, assimilation_config, path_config
from src.assimilation.kalman_nonlinear import NonlinearAssimilator
from src.utils.logging_config import get_logger
import numpy as np

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kalman filter assimilation in AE latent space"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=assimilation_config.beta,
        help=f"observation error scale (default: {assimilation_config.beta})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = get_storage()

    logger.info("=" * 50)
    logger.info("DATA ASSIMILATION — Nonlinear (AE)")
    logger.info("=" * 50)
    logger.info(f"beta       : {args.beta}")
    logger.info(f"background : {data_config.background_path}")
    logger.info(f"obs        : {data_config.obs_path}")

    # load AE artifacts from encoding stage
    mean_map = storage.load_array(path_config.mean_map)
    std_map = storage.load_array(path_config.std_map)
    latent_dim = int(np.load(path_config.ae_latent)[0])

    assimilator = NonlinearAssimilator(
        weights_path=path_config.ae_weights,
        mean_map=mean_map,
        std_map=std_map,
        latent_dim=latent_dim,
        beta=args.beta,
    )

    results = assimilator.run(
        background_path=data_config.background_path,
        obs_path=data_config.obs_path,
        truth_path=data_config.test_path,
    )

    storage.save_results(results, "assimilation_nonlinear_results")

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"background MSE : {results['mse_background']:.3e}")
    logger.info(f"analysis MSE   : {results['mse_physical']:.3e}")
    logger.info(
        f"improvement    : "
        f"{((results['mse_background'] - results['mse_physical']) / results['mse_background'] * 100):.1f}%"
    )
    logger.info(f"encoding time  : {results['encoding_time_s']}s")
    logger.info(f"latent update  : {results['latent_update_time_s']}s")
    logger.info(f"decoding time  : {results['decoding_time_s']}s")


if __name__ == "__main__":
    main()
