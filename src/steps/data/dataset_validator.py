import os
import tqdm
import cv2
from minio.error import S3Error
from zenml.logger import get_logger
from zenml import step

# Expected file extensions
image_extensions = ".png"
annotation_extensions = ".png"
label_extensions = ".txt"


def get_filenames_without_extension(dir_path, extension):
    return {
        os.path.splitext(f)[0] for f in os.listdir(dir_path) if f.endswith(extension)
    }


@step
def dataset_validator(path_dir: str) -> str:
    """
    Validates the dataset by checking if the annotations and labels are consistent with the images.

    Args:
        path_dir (str): The path to the dataset directory.
    """
    # add the path to the dataset
    images_dir = os.path.join(path_dir, "images")
    annotations_dir = os.path.join(path_dir, "annotations")
    labels_dir = os.path.join(path_dir, "labels")

    # Get sets of filenames without their extensions
    image_files = get_filenames_without_extension(images_dir, image_extensions)
    annotation_files = get_filenames_without_extension(
        annotations_dir, annotation_extensions
    )
    label_files = get_filenames_without_extension(labels_dir, label_extensions)

    # Initialize validation flags
    all_files_valid = True

    # Check for corresponding annotation and label for each image
    missing_annotations = image_files - annotation_files
    missing_labels = image_files - label_files

    if missing_annotations:
        all_files_valid = False
        for missing_annotation in missing_annotations:
            print(f"Missing annotation for image: {missing_annotation}")

    if missing_labels:
        all_files_valid = False
        for missing_label in missing_labels:
            print(f"Missing label for image: {missing_label}")

    # Check if the numbers of files are equal
    if not (len(image_files) == len(annotation_files) == len(label_files)):
        all_files_valid = False
        print(
            "The number of files in images, annotations, and labels folders are not equal."
        )

    # Output validation result
    if all_files_valid:
        print("All files are valid and consistent.")
    else:
        print("Some files are missing or inconsistent.")

    return path_dir
