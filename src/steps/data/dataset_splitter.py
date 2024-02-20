import os
import random
import shutil
from zenml import step
from tqdm import tqdm

# make a function that takes a path to a dataset, it has two folders (images, labels) that have files with the same name but different extensions.
# The function should split the dataset into train, val and test datasets, and make sure that the images and labels are in the same order, and put everything in a new folder called custom_dataset.
# use 80% of the data for training, 10% for validation, and 10% for testing.
# The function should return the path to the new folder.

# The function should be used in the gitflow_experiment_pipeline function in the dataset_splitter step.

import json
import os


def generate_config_yaml(dataset_path: str) -> None:
    """
    Generates a config.yaml file with paths to train, val, test datasets and class names.

    Args:
        dataset_path: The path to the dataset folder.
        percentage: The percentage of data to put in the custom dataset.
    """
    
    label_map_path = os.path.join(dataset_path, "label_map.json")

    # Load class names from label_map.json
    with open(label_map_path, "r") as file:
        class_names = json.load(file)  # Directly use this as the class names mapping

    config_content = {
        "path": "human_parsing_dataset/custom_dataset",  # Relative path to dataset
        "train": "train/images",  # Relative path to train images
        "val": "val/images",  # Relative path to val images
        "test": "test/images",  # Relative path to test images
        "names": class_names,
    }

    # Generate config.yaml content
    config_lines = [
        "# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]",
        f"path: {config_content['path']} # dataset root dir",
        f"train: {config_content['train']} # train images (relative to 'path')",
        f"val: {config_content['val']} # val images (relative to 'path')",
        f"test: {config_content['test']} # test images (relative to 'path')",
        "",
        "# Classes",
        "names:",
    ]

    for index, name in config_content["names"].items():
        config_lines.append(f"  {index}: {name}")

    # Write to config.yaml
    config_path = os.path.join(dataset_path + "/custom_dataset", "config.yaml")
    with open(config_path, "w") as file:
        file.write("\n".join(config_lines))

    print(f"Generated config.yaml at {config_path}")


@step
def dataset_splitter(dataset_path: str, percentage = 1.0) -> str:
    """
    Split the dataset into train, val and test datasets, and make sure that the images and labels are in the same order, and put everything in a new folder called custom_dataset.

    Args:
        dataset_path: The path to the dataset folder.
        percentage: The percentage of data to put in the custom dataset.

    Returns:
        The path to the new folder.
    """

    # check if the dataset_path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"The dataset path {dataset_path} does not exist.")

    # check if the dataset_path has the images and labels folders
    if not os.path.exists(os.path.join(dataset_path, "images")):
        raise ValueError(
            f"The dataset path {dataset_path} does not have an images folder."
        )

    if not os.path.exists(os.path.join(dataset_path, "labels")):
        raise ValueError(
            f"The dataset path {dataset_path} does not have a labels folder."
        )

    # check if the custom_dataset folder already exists
    if os.path.exists(os.path.join(dataset_path, "custom_dataset")):
        return os.path.join(dataset_path, "custom_dataset")

    # Create the new folder for the custom dataset
    custom_dataset_path = os.path.join(dataset_path, "custom_dataset")
    os.makedirs(custom_dataset_path, exist_ok=True)

    # Get the list of image and label files
    image_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(dataset_path, "images"))
            if f.endswith(".png")
        ]
    )
    label_files = sorted(
        [
            f
            for f in os.listdir(os.path.join(dataset_path, "labels"))
            if f.endswith(".txt")
        ]
    )
    
        # Use only a percentage of the dataset
    dataset_size = int(len(image_files) * percentage)
    image_files = image_files[:dataset_size]
    label_files = label_files[:dataset_size]

    # Calculate the number of samples for each split
    total_samples = len(image_files)
    train_samples = int(total_samples * 0.8)
    val_samples = int(total_samples * 0.1)
    test_samples = total_samples - train_samples - val_samples

    # Shuffle the image and label files
    combined_files = list(zip(image_files, label_files))
    random.shuffle(combined_files)
    image_files, label_files = zip(*combined_files)

    # Directories to be created for the splits
    split_dirs = ["train", "val", "test"]

    for split_dir in split_dirs:
        # Adjusted to create 'images' and 'labels' subdirectories inside each split directory
        os.makedirs(
            os.path.join(custom_dataset_path, split_dir, "images"), exist_ok=True
        )
        os.makedirs(
            os.path.join(custom_dataset_path, split_dir, "labels"), exist_ok=True
        )

    # Adjusted file copying process with tqdm progress bar
    print("Copying files to train split...")
    for i in tqdm(range(train_samples), desc="Train Split"):
        shutil.copy(
            os.path.join(dataset_path, "images", image_files[i]),
            os.path.join(custom_dataset_path, "train", "images", image_files[i]),
        )
        shutil.copy(
            os.path.join(dataset_path, "labels", label_files[i]),
            os.path.join(custom_dataset_path, "train", "labels", label_files[i]),
        )

    print("\nCopying files to validation split...")
    for i in tqdm(
        range(train_samples, train_samples + val_samples), desc="Validation Split"
    ):
        shutil.copy(
            os.path.join(dataset_path, "images", image_files[i]),
            os.path.join(custom_dataset_path, "val", "images", image_files[i]),
        )
        shutil.copy(
            os.path.join(dataset_path, "labels", label_files[i]),
            os.path.join(custom_dataset_path, "val", "labels", label_files[i]),
        )

    print("\nCopying files to test split...")
    for i in tqdm(range(train_samples + val_samples, total_samples), desc="Test Split"):
        shutil.copy(
            os.path.join(dataset_path, "images", image_files[i]),
            os.path.join(custom_dataset_path, "test", "images", image_files[i]),
        )
        shutil.copy(
            os.path.join(dataset_path, "labels", label_files[i]),
            os.path.join(custom_dataset_path, "test", "labels", label_files[i]),
        )

    print(
        f"\nDataset split into train, val, and test datasets in {custom_dataset_path}"
    )

    # Generate config.yaml
    generate_config_yaml(dataset_path)

    return custom_dataset_path
