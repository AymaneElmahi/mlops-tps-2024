import os
import cv2
from typing import List
from zenml.logger import get_logger
from zenml.steps import BaseStep
from zenml import step
import json
import tqdm

def mask_to_yolo(mask_path, label_map, output_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    for pixel_value in range(1, len(label_map)):
        pixels_equal_to = mask == pixel_value
        contours, hierarchy = cv2.findContours(pixels_equal_to.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            polygon_contour = max_contour.squeeze()
            polygon_contour = polygon_contour.tolist()
            polygon_contour = [[x, y] for x, y in polygon_contour]
            normalized_polygon_contour = [[x / mask.shape[1], y / mask.shape[0]] for x, y in polygon_contour]
            
            # check if contour is significant enough to be considered
            if len(polygon_contour) > 2:
                with open(output_path, 'a') as f:  # Open the file in append mode
                    f.write(f'{pixel_value} ')
                    for x, y in normalized_polygon_contour:
                        f.write(f'{x} {y} ')
                    f.write('\n')
                    

def process_image(image_file, images_dir, output_dir, label_map):
    '''
    Process an image and its mask to YOLO format.
    '''
    mask_path = os.path.join(images_dir, image_file)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.txt")
    mask_to_yolo(mask_path, label_map, output_path)
    

@step
def dataset_to_yolo_converter(path_dir: str) -> str:
    '''
    Process all images in a directory to YOLO format.
    '''
    
    logger = get_logger(__name__)

    # check if labels folder exists and is not empty, if it is, return
    if os.path.exists(os.path.join(path_dir, 'labels')) and os.listdir(os.path.join(path_dir, 'labels')):
        logger.info('Dataset already converted to YOLO format')
        return path_dir

    # if a folder named labels doesn't exist, create it
    output_dir = os.path.join(path_dir, 'labels')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # check for label_map.json and load it, if it doesn't exist, give a warning
    label_map_path = os.path.join(path_dir, 'label_map.json')
    if os.path.exists(label_map_path):
        label_map = json.load(open(label_map_path))
    else:
        logger.warning('label_map.json not found')
        return
    
    # check if a folder named annotations exists, if it doesn't, give a warning
    if not os.path.exists(os.path.join(path_dir, 'annotations')):
        logger.error('annotations folder not found')
        return  
    
    annotations_dir = os.path.join(path_dir, 'annotations')
    
    # use tqdm to show progress bar
    total_images = len(os.listdir(annotations_dir))
    for image_file in tqdm.tqdm(os.listdir(annotations_dir), total=total_images, desc="Processing images"):
        process_image(image_file, annotations_dir, output_dir, label_map)
        
    return path_dir
