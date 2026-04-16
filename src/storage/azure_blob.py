import io
import os
import numpy as np
from joblib import dump, load

from src.storage.base import BaseStorage
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AzureStorage(BaseStorage):
    # Azure implementation of the same storage interface.
    # The pipeline doesn't know or care that this exists -
    # it just calls load_array() and save_results() the same way.
    #
    # To activate:
    #   export ENVIRONMENT=azure
    #   export AZURE_STORAGE_CONNECTION_STRING=...
    #   export COSMOS_CONNECTION_STRING=...

    def __init__(
        self,
        container_name: str,
        cosmos_db: str,
        cosmos_container: str
    ):
        # lazy imports so local runs don't need azure packages
        from azure.storage.blob import BlobServiceClient
        from azure.cosmos import CosmosClient

        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        cosmos_str = os.environ.get("COSMOS_CONNECTION_STRING")

        if not conn_str:
            raise EnvironmentError(
                "AZURE_STORAGE_CONNECTION_STRING not set"
            )
        if not cosmos_str:
            raise EnvironmentError(
                "COSMOS_CONNECTION_STRING not set"
            )

        self.blob_client = BlobServiceClient.from_connection_string(
            conn_str
        )
        self.container = container_name

        cosmos_client = CosmosClient.from_connection_string(cosmos_str)
        db = cosmos_client.get_database_client(cosmos_db)
        self.cosmos_container = db.get_container_client(cosmos_container)

        logger.info("azure storage clients initialised")

    def _get_blob(self, path: str):
        return self.blob_client.get_container_client(self.container).get_blob_client(path)

    def load_array(self, path: str) -> np.ndarray:
        logger.info(f"downloading blob: {path}")
        data = self._get_blob(path).download_blob().readall()
        return np.load(io.BytesIO(data))

    def save_array(self, array: np.ndarray, path: str) -> None:
        buf = io.BytesIO()
        np.save(buf, array)
        buf.seek(0)
        self._get_blob(path).upload_blob(buf, overwrite=True)
        logger.info(f"uploaded array to blob: {path}")

    def save_model(self, model: object, path: str) -> None:
        buf = io.BytesIO()
        dump(model, buf)
        buf.seek(0)
        self._get_blob(path).upload_blob(buf, overwrite=True)
        logger.info(f"uploaded model to blob: {path}")

    def load_model(self, path: str) -> object:
        logger.info(f"downloading model from blob: {path}")
        data = self._get_blob(path).download_blob().readall()
        return load(io.BytesIO(data))

    def save_results(self, results: dict, run_id: str) -> None:
        # goes straight to Cosmos DB as a JSON document
        self.cosmos_container.upsert_item({
            "id": run_id,
            **results
        })
        logger.info(f"results written to cosmos: {run_id}")