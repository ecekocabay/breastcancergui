import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage import io
import os


def compute_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    if image.max() > 1:
        image = (image / image.max()).astype(np.float32)
    hog_features = hog(image,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm='L2-Hys',
                       visualize=False,
                       feature_vector=True)
    return hog_features


def extract_hog_features(image_folder, output_csv, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"The folder '{image_folder}' does not exist.")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.pgm', '.png', '.jpg'))]
    if not image_files:
        raise FileNotFoundError(f"No images found in '{image_folder}' with supported formats (.pgm, .png, .jpg).")

    # Initialize the output CSV file with headers
    with open(output_csv, 'w') as f:
        columns = ["Image Name"] + [f"Feature_{i + 1}" for i in range(36)]  # Replace 36 with the actual feature length
        f.write(",".join(columns) + "\n")

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            image = io.imread(image_path, as_gray=True)
            hog_features = compute_hog(image, pixels_per_cell, cells_per_block, orientations)

            # Append to the CSV file
            with open(output_csv, 'a') as f:
                f.write(f"{image_file}," + ",".join(map(str, hog_features)) + "\n")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    print(f"Feature matrix successfully saved to '{output_csv}'.")


if __name__ == "__main__":
    image_folder = r'/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/data/images'   # Update to the correct path
    output_csv = r'/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/data/hog_feature_matrix.csv'

    try:
        extract_hog_features(image_folder, output_csv)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")