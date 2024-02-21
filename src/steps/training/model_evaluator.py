from zenml import step
import os
import torch
from ultralytics import YOLO


@step
def model_evaluator(
    model_path: str,
    dataset_path: str,
) -> dict:
    """
    Evaluate the trained model.

    Args:
        model_path: The path where the model is saved.
        dataset_path: The path to the dataset used for evaluation.

    Returns:
        A dictionary containing evaluation metrics.
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).to(device)

    # Prepare the dataset
    data = dataset_path + "/config.yaml"

    # Run model evaluation
    results = model.evaluate(
        data=data,
        imgsz=640,  # Assuming you have this as a standard evaluation image size
        batch_size=16,  # You might adjust this based on your hardware
        iou_thres=0.5,  # IOU threshold for mAP calculation
        conf_thres=0.5,  # Confidence threshold for predictions
        task="val",  # Specifies the task is evaluation/validation
    )

    # Extract evaluation metrics
    metrics = {
        "precision": results.precision.mean(),
        "recall": results.recall.mean(),
        "mAP_0.5": results.map50,
    }

    print(f"Evaluation metrics: {metrics}")

    return metrics
