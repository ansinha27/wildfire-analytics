from abc import ABC, abstractmethod
import numpy as np


class BaseStorage(ABC):
    # I wanted the pipeline to be completely independent
    # of where the data actually lives - local disk or Azure.
    # So everything goes through this interface.
    # Swap the implementation in config.py, nothing else changes.

    @abstractmethod
    def load_array(self, path: str) -> np.ndarray:
        # load a numpy array - could be local or from blob storage
        pass

    @abstractmethod
    def save_array(self, array: np.ndarray, path: str) -> None:
        # persist an array - locally or to blob
        pass

    @abstractmethod
    def save_model(self, model: object, path: str) -> None:
        # save a fitted model artifact
        pass

    @abstractmethod
    def load_model(self, path: str) -> object:
        # reload a previously saved model
        pass

    @abstractmethod
    def save_results(self, results: dict, run_id: str) -> None:
        # locally this writes a json file
        # in Azure this goes straight to Cosmos DB
        pass