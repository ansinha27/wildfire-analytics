# pipeline.py
#
# This is the main entry point for the whole analytics pipeline.
# Each stage is independent - you can run them individually
# via scripts/ or run the full pipeline end to end from here.
#
# Usage:
#   python pipeline.py                        # run all stages
#   python pipeline.py --stage compression    # run one stage
#   python pipeline.py --skip encoding        # skip a stage
#
# To run in Azure:
#   export ENVIRONMENT=azure
#   python pipeline.py

import argparse
import sys
from datetime import datetime

from config import (
    get_storage,
    data_config,
    compression_config,
    ae_config,
    assimilation_config,
    path_config
)
from src.utils.logging_config import get_logger
from src.utils.metrics import Timer

logger = get_logger(__name__)

# these are the stages in the order they need to run
# each stage depends on the previous one's outputs
STAGES = [
    "compression",
    "encoding",
    "fusion",
    "assimilation_linear",
    "assimilation_nonlinear"
]


def run_compression(storage):
    # Task 1: fit TruncatedSVD on training data,
    # evaluate reconstruction MSE on test data
    logger.info("=" * 50)
    logger.info("STAGE: Linear Compression (TSVD)")
    logger.info("=" * 50)

    from src.compression.tsvd import TSVDCompressor

    with Timer("compression fit") as t:
        compressor = TSVDCompressor(
            n_components=compression_config.n_components,
            batch_size=compression_config.batch_size,
            n_iter=compression_config.n_iter
        )
        compressor.fit(data_config.train_path)

    storage.save_model(compressor.model, path_config.tsvd_model)
    storage.save_array(compressor.mean, path_config.mean_train)

    mse, recon_time = compressor.evaluate(data_config.test_path)

    return {
        "stage": "compression",
        "fit_time_s": t.elapsed,
        "reconstruction_time_s": recon_time,
        "mse": mse,
        "n_components": compression_config.n_components
    }


def run_encoding(storage):
    # Task 2: train UNet autoencoder, evaluate on test data
    logger.info("=" * 50)
    logger.info("STAGE: Nonlinear Compression (Autoencoder)")
    logger.info("=" * 50)

    from src.compression.autoencoder import AutoencoderCompressor

    with Timer("autoencoder training") as t:
        encoder = AutoencoderCompressor(
            latent_dim=ae_config.latent_dim,
            batch_size=ae_config.batch_size,
            n_epochs=ae_config.n_epochs,
            lr=ae_config.lr,
            alpha_mse=ae_config.alpha_mse,
            alpha_edge=ae_config.alpha_edge,
            patience=ae_config.patience
        )
        encoder.fit(data_config.train_path)

    storage.save_model(encoder, path_config.ae_weights)

    mse, recon_time = encoder.evaluate(data_config.test_path)

    return {
        "stage": "encoding",
        "training_time_s": t.elapsed,
        "reconstruction_time_s": recon_time,
        "mse": mse,
        "latent_dim": ae_config.latent_dim
    }


def run_fusion(storage):
    # Task 3: fuse two satellite observations
    # in the TSVD latent space
    logger.info("=" * 50)
    logger.info("STAGE: Data Fusion")
    logger.info("=" * 50)

    from src.fusion.latent_fusion import LatentFusion

    compressor = storage.load_model(path_config.tsvd_model)
    mean = storage.load_array(path_config.mean_train)

    with Timer("fusion") as t:
        fusion = LatentFusion(compressor=compressor, mean=mean)
        result = fusion.fuse(
            obs_path=data_config.obs_path,
            idx1=0,
            idx2=-1
        )

    return {
        "stage": "fusion",
        "fusion_time_s": t.elapsed,
        "method": "tsvd_latent_average"
    }


def run_assimilation_linear(storage):
    # Task 4: Kalman filter in TSVD latent space
    # blends background forecast with satellite observations
    logger.info("=" * 50)
    logger.info("STAGE: Data Assimilation (Linear / TSVD)")
    logger.info("=" * 50)

    from src.assimilation.kalman_linear import LinearAssimilator

    compressor = storage.load_model(path_config.tsvd_model)
    mean = storage.load_array(path_config.mean_train)

    assimilator = LinearAssimilator(
        compressor=compressor,
        mean=mean,
        beta=assimilation_config.beta
    )

    with Timer("linear assimilation") as t:
        results = assimilator.run(
            background_path=data_config.background_path,
            obs_path=data_config.obs_path,
            truth_path=data_config.test_path
        )

    return {
        "stage": "assimilation_linear",
        "total_time_s": t.elapsed,
        **results
    }


def run_assimilation_nonlinear(storage):
    # Task 5: same Kalman filter approach but in the
    # autoencoder latent space instead of TSVD space
    # generally gives better MSE because the AE latent
    # space captures nonlinear structure
    logger.info("=" * 50)
    logger.info("STAGE: Data Assimilation (Nonlinear / AE)")
    logger.info("=" * 50)

    from src.assimilation.kalman_nonlinear import NonlinearAssimilator
    import numpy as np

    mean_map = storage.load_array(path_config.mean_map)
    std_map = storage.load_array(path_config.std_map)
    latent_dim = int(np.load(path_config.ae_latent)[0])

    assimilator = NonlinearAssimilator(
        weights_path=path_config.ae_weights,
        mean_map=mean_map,
        std_map=std_map,
        latent_dim=latent_dim,
        beta=assimilation_config.beta
    )

    with Timer("nonlinear assimilation") as t:
        results = assimilator.run(
            background_path=data_config.background_path,
            obs_path=data_config.obs_path,
            truth_path=data_config.test_path
        )

    return {
        "stage": "assimilation_nonlinear",
        "total_time_s": t.elapsed,
        **results
    }


# maps stage names to their functions
# makes it easy to add new stages later
STAGE_RUNNERS = {
    "compression":            run_compression,
    "encoding":               run_encoding,
    "fusion":                 run_fusion,
    "assimilation_linear":    run_assimilation_linear,
    "assimilation_nonlinear": run_assimilation_nonlinear
}


def run_pipeline(stages: list, storage):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"run_id": run_id, "stages": {}}

    logger.info(f"pipeline starting | run_id: {run_id}")
    logger.info(f"stages to run: {stages}")

    for stage in stages:
        try:
            result = STAGE_RUNNERS[stage](storage)
            all_results["stages"][stage] = result
            logger.info(f"stage complete: {stage}")

        except Exception as e:
            logger.error(f"stage failed: {stage} | error: {e}")
            all_results["stages"][stage] = {
                "stage": stage,
                "status": "failed",
                "error": str(e)
            }
            # stop the pipeline if a stage fails
            # downstream stages depend on earlier outputs
            raise

    storage.save_results(all_results, run_id)
    logger.info(f"pipeline complete | run_id: {run_id}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ferguson Wildfire Analytics Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=None,
        help="run a single stage only"
    )
    parser.add_argument(
        "--skip",
        choices=STAGES,
        nargs="+",
        default=[],
        help="skip one or more stages"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    storage = get_storage()

    if args.stage:
        # run just one stage
        stages = [args.stage]
    else:
        # run all stages, minus any skipped ones
        stages = [s for s in STAGES if s not in args.skip]

    run_pipeline(stages, storage)