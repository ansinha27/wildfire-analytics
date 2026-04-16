# run_fusion.py
#
# Fuses two satellite observations in TSVD latent space.
# Requires compression stage to have run first.
#
# Usage:
#   python scripts/run_fusion.py
#   python scripts/run_fusion.py --idx1 0 --idx2 10

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import get_storage, data_config, path_config
from src.fusion.latent_fusion import LatentFusion
from src.utils.logging_config import get_logger
from src.utils.metrics import Timer
from src.utils.visualisation import plot_fusion_result

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Data fusion in TSVD latent space")
    parser.add_argument(
        "--idx1", type=int, default=0, help="index of first observation frame"
    )
    parser.add_argument(
        "--idx2", type=int, default=-1, help="index of second observation frame"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for binary mask (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = get_storage()

    logger.info("=" * 50)
    logger.info("DATA FUSION — Latent Space")
    logger.info("=" * 50)
    logger.info(f"obs frame 1 : {args.idx1}")
    logger.info(f"obs frame 2 : {args.idx2}")
    logger.info(f"threshold   : {args.threshold}")

    # load compression artifacts from previous stage
    compressor = storage.load_model(path_config.tsvd_model)
    mean = storage.load_array(path_config.mean_train)

    fusion = LatentFusion(compressor=compressor, mean=mean)

    with Timer("data fusion") as t:
        result = fusion.fuse(
            obs_path=data_config.obs_path,
            idx1=args.idx1,
            idx2=args.idx2,
            threshold=args.threshold,
        )

    # save visualisation
    plot_fusion_result(
        obs1=result["obs1"],
        obs2=result["obs2"],
        fused=result["fused"],
        binary=result["binary"],
    )

    results = {
        "stage": "fusion",
        "idx1": args.idx1,
        "idx2": args.idx2,
        "fusion_time_s": t.elapsed,
        "method": "tsvd_latent_average",
    }
    storage.save_results(results, "fusion_results")

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"fusion time : {t.elapsed}s")
    logger.info("figure saved to outputs/figures/")


if __name__ == "__main__":
    main()
