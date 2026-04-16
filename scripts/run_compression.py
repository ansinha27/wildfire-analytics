# run_compression.py
#
# Fits TruncatedSVD on training data and evaluates
# reconstruction MSE on test data.
#
# Usage:
#   python scripts/run_compression.py
#   python scripts/run_compression.py --n_components 100

import argparse
import sys
from pathlib import Path

# make sure project root is on the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_storage, data_config, compression_config, path_config
from src.compression.tsvd import TSVDCompressor
from src.utils.logging_config import get_logger
from src.utils.metrics import Timer
from src.utils.visualisation import plot_cumulative_variance

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear compression using TruncatedSVD"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=compression_config.n_components,
        help=f"number of SVD modes (default: {compression_config.n_components})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=compression_config.batch_size,
        help="batch size for memory-mapped loading",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = get_storage()

    logger.info("=" * 50)
    logger.info("LINEAR COMPRESSION - TruncatedSVD")
    logger.info("=" * 50)
    logger.info(f"n_components : {args.n_components}")
    logger.info(f"batch_size   : {args.batch_size}")
    logger.info(f"train data   : {data_config.train_path}")
    logger.info(f"test data    : {data_config.test_path}")

    # fit
    compressor = TSVDCompressor(
        n_components=args.n_components,
        batch_size=args.batch_size,
        n_iter=compression_config.n_iter,
    )

    with Timer("full compression pipeline") as t:
        compressor.fit(data_config.train_path)

    # save model artifacts
    storage.save_model(compressor.model, path_config.tsvd_model)
    storage.save_array(compressor.mean, path_config.mean_train)
    logger.info("model artifacts saved")

    # evaluate on test set
    mse, recon_time = compressor.evaluate(data_config.test_path)

    # save results
    results = {
        "stage": "compression",
        "n_components": args.n_components,
        "fit_time_s": t.elapsed,
        "reconstruction_time_s": recon_time,
        "mse": mse,
    }
    storage.save_results(results, "compression_results")

    # summary
    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"fit time          : {t.elapsed}s")
    logger.info(f"reconstruction MSE: {mse:.3e}")
    logger.info(f"reconstruction time: {recon_time}s")


if __name__ == "__main__":
    main()
