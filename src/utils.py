"""
Utility functions for the handwritten digit classifier
"""

import numpy as np
import cv2
from PIL import Image


def preprocess_image(image_data):
    """
    Preprocess an image to match MNIST format (28x28 grayscale)
    
    Args:
        image_data: numpy array of the image
        
    Returns:
        Preprocessed 28x28 numpy array, or None if processing fails
    """
    # Convert to grayscale if needed
    if len(image_data.shape) == 3:
        if image_data.shape[2] == 4:  # RGBA
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    
    # Invert colors if needed (MNIST has white digits on black background)
    # Check if background is lighter than foreground
    if np.mean(image_data) > 127:
        image_data = 255 - image_data
    
    # Apply threshold to clean up the image
    _, image_data = cv2.threshold(image_data, 50, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of the digit
    coords = cv2.findNonZero(image_data)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image_data.shape[1] - x, w + 2 * padding)
    h = min(image_data.shape[0] - y, h + 2 * padding)
    
    # Crop to bounding box
    cropped = image_data[y:y+h, x:x+w]
    
    # Resize to 20x20 maintaining aspect ratio
    if h > w:
        new_h = 20
        new_w = max(1, int(20 * w / h))
    else:
        new_w = 20
        new_h = max(1, int(20 * h / w))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 image with digit centered
    final_image = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize
    final_image = final_image.astype(np.float32)
    
    return final_image


def load_image(image_path):
    """
    Load an image from file path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of the image
    """
    image = Image.open(image_path)
    return np.array(image)


def visualize_preprocessing(original_image, processed_image):
    """
    Create a visualization showing original and processed images side by side
    
    Args:
        original_image: Original image array
        processed_image: Processed 28x28 image array
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    if len(original_image.shape) == 3:
        ax1.imshow(original_image)
    else:
        ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Processed image
    ax2.imshow(processed_image, cmap='gray')
    ax2.set_title('Processed (28x28)')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def predict_digit(model, scaler, image_array):
    """
    Predict digit from image array
    
    Args:
        model: Trained KNN model
        scaler: Fitted StandardScaler
        image_array: Image as numpy array
        
    Returns:
        tuple: (predicted_digit, confidence, all_probabilities)
    """
    # Preprocess
    processed = preprocess_image(image_array)
    if processed is None:
        return None, None, None
    
    # Flatten and scale
    image_vector = processed.flatten().reshape(1, -1)
    image_scaled = scaler.transform(image_vector)
    
    # Predict
    prediction = model.predict(image_scaled)[0]
    probabilities = model.predict_proba(image_scaled)[0]
    confidence = probabilities[prediction]
    
    return int(prediction), float(confidence), probabilities


def get_top_k_predictions(probabilities, k=3):
    """
    Get top K predictions with their probabilities
    
    Args:
        probabilities: Array of probabilities for each class
        k: Number of top predictions to return
        
    Returns:
        list of tuples: [(digit, probability), ...]
    """
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    return [(int(idx), float(probabilities[idx])) for idx in top_k_indices]
