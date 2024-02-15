"""
This module contains functions for extracting data sources from a Minio bucket
and saving them to a local extraction path.
"""

import os
import tqdm
from minio.error import S3Error
from minio import Minio

from hydra.utils import to_absolute_path
from zenml.logger import get_logger
from zenml import step


from src.config.settings import (
    MINIO_DATA_SOURCES_BUCKET_NAME,
    MINIO_ENDPOINT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    EXTRACTED_DATASETS_PATH,
)


def get_minio_client() -> Minio:
    """
    Returns a Minio client to the endpoint.
    """
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )


@step
def data_source_extractor(
    data_source: str,
    bucket_name: str = MINIO_DATA_SOURCES_BUCKET_NAME,
    extraction_path: str = to_absolute_path(EXTRACTED_DATASETS_PATH),
) -> None:
    """
    Extracts the data source from the bucket to the extraction path.

    Args:
        data_source (str): The name of the data source to extract.
        bucket_name (str, optional): The name of the Minio bucket.
        Defaults to MINIO_DATA_SOURCES_BUCKET_NAME.
        extraction_path (str, optional): The local path to save the extracted data source.
        Defaults to EXTRACTED_DATASETS_PATH.
    """
    logger = get_logger(__name__)
    minio_client = get_minio_client()

    try:
        # Check if the bucket exists
        if not minio_client.bucket_exists(bucket_name):
            logger.error(f"The bucket {bucket_name} does not exist.")
            raise ValueError(f"The bucket {bucket_name} does not exist.")

        # Check if the dataset has already been downloaded, check if its not empty
        if os.path.exists(extraction_path):
            if os.listdir(extraction_path):
                logger.info(
                    f"The data source {data_source} has already been extracted."
                )
                return

        # List the objects in the data source
        objects = minio_client.list_objects(bucket_name, data_source, recursive=True)

        # Extract the objects to the extraction path
        for obj in tqdm.tqdm(objects, desc="Extracting data source"):
            if not obj.is_dir:
                object_name = obj.object_name
                local_path = os.path.join(extraction_path, object_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Download the object
                minio_client.fget_object(bucket_name, object_name, local_path)

    except S3Error as e:
        logger.error(f"An error occurred while extracting the data source: {e}")
        raise e

    logger.info(f"The data source {data_source} has been extracted.")
    return
