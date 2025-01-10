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