import joblib
from skimage.feature import local_binary_pattern
from skimage import io
from skimage.transform import resize
import numpy as np
from segmentation import process_image  # Import the segmentation function

def compute_lbp(image, radius=1, n_points=8):
    """
    Computes the Local Binary Pattern (LBP) histogram for a given image.

    :param image: Grayscale image array.
    :param radius: Radius for LBP computation.
    :param n_points: Number of points for LBP computation.
    :return: Normalized histogram of LBP features.
    """
    # Normalize and convert to uint8 if necessary
    if image.max() > 1:
        image = (image / image.max() * 255).astype(np.uint8)

    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist

def predict_image(image_path, model_path):
    """
    Predicts the class of a single mammogram image using the trained model.

    :param image_path: Path to the image file.
    :param model_path: Path to the saved model.
    :return: Predicted class label and model confidence.
    """
    # Load the trained model
    rf_model = joblib.load(model_path)

    # Load the pre-saved scaler
    scaler_path = '/Users/ecekocabay/Desktop/CNG491/BreastCancerGUI 2/scaler.pkl'  # Adjust the path as needed
    scaler = joblib.load(scaler_path)

    # Segment the image
    segmented_image = process_image(image_path)

    if segmented_image is None:
        raise ValueError("Segmentation failed. No region of interest detected.")

    # Resize the segmented image for consistency
    segmented_image = resize(segmented_image, (256, 256), anti_aliasing=True)

    # Extract LBP features from the segmented image
    lbp_features = compute_lbp(segmented_image)

    # Normalize LBP features using the preloaded scaler
    normalized_features = scaler.transform([lbp_features])  # Use the scaler to transform

    # Predict the class and confidence
    prediction = rf_model.predict(normalized_features)
    confidence = rf_model.predict_proba(normalized_features).max() * 100  # Get the confidence score
    return prediction[0], confidence