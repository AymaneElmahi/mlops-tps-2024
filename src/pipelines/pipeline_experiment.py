from omegaconf import OmegaConf
from zenml import pipeline

from src.config.settings import EXTRACTED_DATASETS_PATH, MLFLOW_EXPERIMENT_PIPELINE_NAME

from src.config.settings import (
    MINIO_DATA_SOURCES_BUCKET_NAME,
    MINIO_DATASETS_BUCKET_NAME,
    MINIO_ENDPOINT,
    MINIO_PENDING_ANNOTATIONS_BUCKET_NAME,
    MINIO_PENDING_REVIEWS_BUCKET_NAME,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
)

from minio import Minio, S3Error
from minio.commonconfig import ENABLED, CopySource
from minio.datatypes import Object
from minio.helpers import ObjectWriteResult
from minio.versioningconfig import VersioningConfig

# from src.steps.data.data_extractor import dataset_extractor
from src.steps.data.datalake_initializers import (
    data_source_list_initializer,
    minio_client_initializer,
)
from src.steps.data.dataset_preparators import data_source_extractor

from src.steps.data.dataset_to_yolo_converter import (
    dataset_to_yolo_converter,
)

from src.steps.data.dataset_validator import (
    dataset_validator,
)

# from src.steps.training.model_appraisers import model_appraiser
# from src.steps.training.model_evaluators import model_evaluator
# from src.steps.training.model_trainers import (
#     get_pre_trained_weights_path,
#     model_trainer,
# )


@pipeline(name=MLFLOW_EXPERIMENT_PIPELINE_NAME)
def gitflow_experiment_pipeline(cfg: str) -> None:
    """
    Experiment a local training and evaluate if the model can be deployed.

    Args:
        cfg: The Hydra configuration.
    """
    pipeline_config = OmegaConf.to_container(OmegaConf.create(cfg))

    data_source_extractor(
        data_source="human_parsing_dataset",
        bucket_name=MINIO_DATA_SOURCES_BUCKET_NAME,
        extraction_path=EXTRACTED_DATASETS_PATH,
    )

    converted_path = dataset_to_yolo_converter(
        path_dir=EXTRACTED_DATASETS_PATH + "/human_parsing_dataset"
    )

    validated_path = dataset_validator(
        path_dir=converted_path
    )

    # Evaluate the model
    # test_metrics_result = model_evaluator(
    #     ...
    # )
