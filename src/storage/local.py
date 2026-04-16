import json
import numpy as np
from joblib import dump, load
from pathlib import Path

from src.storage.base import BaseStorage
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LocalStorage(BaseStorage):
    # local filesystem implementation
    # this is what runs during development and testing
    # when I'm ready to move to Azure I just swap this
    # out for AzureStorage in config.py

    def load_array(self, path: str) -> np.ndarray:
        logger.info(f"loading array from {path}")
        # memory map by default - avoids loading
        # the full array into RAM on large files
        return np.load(path, mmap_mode="r")

    def save_array(self, array: np.ndarray, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving array to {path}")
        np.save(path, array)

    def save_model(self, model: object, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving model to {path}")
        dump(model, path)

    def load_model(self, path: str) -> object:
        logger.info(f"loading model from {path}")
        return load(path)

    def save_results(self, results: dict, run_id: str) -> None:
        # write results as json so they're human readable
        # and easy to compare across runs
        out = Path("outputs") / f"{run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"results saved to {out}")