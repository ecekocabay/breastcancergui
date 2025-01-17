import os

import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage import io

def compute_lbp(image, radius=1, n_points=8):
    """
    Computes the Local Binary Pattern (LBP) histogram for a given image.

    :param image: Grayscale image array.
    :param radius: Radius for LBP computation.
    :param n_points: Number of points for LBP computation.
    :return: Normalized histogram of LBP features.
    """
    if image.max() > 1:
        image = (image / image.max() * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist

def extract_features(image_folder, output_csv, radius=1, n_points=8):
    """
    Extracts LBP features from all images in the specified folder and saves them to a CSV file.

    :param image_folder: Path to the folder containing images.
    :param output_csv: Path to save the extracted feature matrix as a CSV file.
    :param radius: Radius for LBP computation.
    :param n_points: Number of points for LBP computation.
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('results.jpg')]
    feature_matrix = []
    image_names = []

    for image_file in image_files:
        image = io.imread(os.path.join(image_folder, image_file), as_gray=True)
        lbp_hist = compute_lbp(image, radius=radius, n_points=n_points)
        feature_matrix.append(lbp_hist)
        image_names.append(image_file)

    # Create a DataFrame
    columns = [f"Feature_{i+1}" for i in range(len(feature_matrix[0]))]
    feature_matrix_df = pd.DataFrame(feature_matrix, columns=columns)
    feature_matrix_df.insert(0, "Image Name", image_names)

    # Save to CSV
    feature_matrix_df.to_csv(output_csv, index=False)
    print(f"Feature matrix successfully saved to '{output_csv}'.")

def extract_features_single(image_path, radius=1, n_points=8):
    """
    Extracts LBP features from a single image.

    :param image_path: Path to the input image.
    :param radius: Radius for LBP computation.
    :param n_points: Number of points for LBP computation.
    :return: LBP feature vector as a NumPy array.
    """
    image = io.imread(image_path, as_gray=True)
    lbp_features = compute_lbp(image, radius=radius, n_points=n_points)
    return lbp_features


if __name__ == "__main__":
    # Example usage
    image_folder = r'/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/data/output_images'  # Update with your folder path
    output_csv = r'/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/segmented_data/lbp_feature_matrix_segmented.csv'  # Desired output CSV file name
    extract_features(image_folder, output_csv)