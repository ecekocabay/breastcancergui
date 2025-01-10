import cv2
import numpy as np
from collections import deque

def process_image(image_path):
    """
    Full processing pipeline for an input image: segmentation and ROI extraction.

    :param image_path: Path to the input image.
    :return: Cropped region of interest (ROI) after segmentation.
    """
    # Read the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Could not read the image at {image_path}")

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(original_image)

    # Perform region growing segmentation
    h, w = enhanced_image.shape
    seed_point = (w // 2, h // 2)  # Use the center point as the seed
    segmented_mask = region_growing(enhanced_image, seed_point, threshold=15)

    # Extract the ROI
    contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return original_image[y:y + h, x:x + w]

    return None

def region_growing(image, seed_point, threshold=10):
    """
    Performs region growing segmentation.

    :param image: Grayscale image.
    :param seed_point: Tuple (x, y) for the seed point.
    :param threshold: Intensity threshold for region growing.
    :return: Binary mask of the segmented region.
    """
    h, w = image.shape
    segmented_image = np.zeros((h, w), dtype=np.uint8)
    queue = deque([seed_point])
    seed_intensity = image[seed_point[1], seed_point[0]]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        x, y = queue.popleft()
        if segmented_image[y, x] == 0:
            segmented_image[y, x] = 255
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if segmented_image[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_intensity)) <= threshold:
                        queue.append((nx, ny))
    return segmented_image