from zenml import pipeline

from src.steps.data.data_sources_uploaders import data_sources_uploader
from src.steps.data.datalake_initializers import (
    bucket_name_list_initializer,
    data_source_list_initializer,
    datalake_initializer,
    minio_client_initializer,
)


@pipeline
def gitflow_datalake_pipeline(cfg: str) -> None:
    bucket_client = minio_client_initializer()
    bucket_name_list = bucket_name_list_initializer()

    data_source_list = data_source_list_initializer()

    datalake_initializer(bucket_client=bucket_client, bucket_name_list=bucket_name_list)
    data_sources_uploader(
        bucket_client=bucket_client,
        data_source_list=data_source_list,
        after="datalake_initializer",
    )
