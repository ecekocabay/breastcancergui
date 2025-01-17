import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage import io
from collections import deque
import joblib
import cv2
import numpy as np
from collections import deque

# Segmentation Functions
def contrast_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = cv2.medianBlur(enhanced, 5)
    return enhanced


def morphological_operations(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (1000, 1000))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return cleaned_image


def region_growing(image, seed_point, threshold=8):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    stack = deque([seed_point])
    seed_intensity = image[seed_point[1], seed_point[0]]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        x, y = stack.pop()
        if segmented[y, x] == 0:
            segmented[y, x] = 255
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if segmented[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_intensity)) <= threshold:
                        stack.append((nx, ny))
    return segmented


def contour_extraction(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, 255, 2)
    return contour_image, contours


def crop_image_with_contours(original_image, contours):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_region = original_image[y:y+h, x:x+w]
        return cropped_region
    return original_image



def segment_image(filepath):
    """
    Perform segmentation on the input image.

    :param filepath: Path to the uploaded image file.
    :return: Segmented image as a NumPy array.
    """
    # Read the image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {filepath}")
        return None

    # Step 1: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Step 2: Perform region growing for segmentation
    seed_point = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
    segmented = region_growing(enhanced, seed_point, threshold=15)

    # Step 3: Refine segmentation with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    refined = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)

    return refined

