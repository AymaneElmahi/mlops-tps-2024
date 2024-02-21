from src.config.settings import (
    YOLO_PRE_TRAINED_WEIGHTS_NAME,
    YOLO_PRE_TRAINED_WEIGHTS_URL,
)

import os

from ultralytics import YOLO
from zenml import step

import torch


def get_pretrained_weights_path(model_path: str) -> str:
    """
    Get the pre-trained weights path.

    Args:
        model_path: The model path.

    Returns:
        The pre-trained weights path.
    """
    pre_trained_weights_path = os.path.join(model_path, YOLO_PRE_TRAINED_WEIGHTS_NAME)
    if not os.path.exists(pre_trained_weights_path):
        os.system(f"wget -O {pre_trained_weights_path} {YOLO_PRE_TRAINED_WEIGHTS_URL}")
    return pre_trained_weights_path


@step
def model_trainer(
    model_path: str,
    dataset_path: str,
    pipeline_config: dict,
) -> str:
    """
    Train the model.

    Args:
        model_path: The model path.
        dataset_path: The dataset path.
        pipeline_config: The pipeline configuration.

    Returns:
        The model path.
    """

    pre_trained_weights_path = get_pretrained_weights_path(model_path)

    data = dataset_path + "/config.yaml"

    # check if the device is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = YOLO(pre_trained_weights_path).to(device)

    print("Starting training...")

    model.train(
        data=data,
        epochs=pipeline_config["model"]["epochs"],
        batch=pipeline_config["model"]["batch_size"],
        imgsz=pipeline_config["model"]["imgsz"],
        project=model_path,
    )

    print(f"Trained model at {model_path}")

    return model_path
