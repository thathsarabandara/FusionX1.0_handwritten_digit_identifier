"""
Handwritten Digit Classifier - Streamlit Demo Application
Uses K-Nearest Neighbors (KNN) to classify handwritten digits
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🔢 Handwritten Digit Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Draw a digit and let AI recognize it using K-Nearest Neighbors!</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses **K-Nearest Neighbors (KNN)** algorithm to classify handwritten digits (0-9).
    
    **How it works:**
    1. Draw a digit on the canvas
    2. The model preprocesses your drawing
    3. KNN finds similar digits in training data
    4. Returns the most common digit among neighbors
    """)
    
    st.header("🎯 Instructions")
    st.write("""
    1. **Draw** a digit (0-9) on the canvas
    2. Try to center your digit
    3. Click **Predict** to classify
    4. Click **Clear** to start over
    """)
    
    st.header("⚙️ Settings")
    stroke_width = st.slider("Brush Size", 1, 25, 15)
    
    st.header("📊 Model Info")
    if os.path.exists('models/knn_digit_classifier.pkl'):
        st.success("✅ Model loaded")
        st.info("Algorithm: K-Nearest Neighbors")
    else:
        st.error("❌ Model not found")
        st.warning("Please train the model first using the notebook.")

# Load model and scaler
@st.cache_resource
def load_model():
    """Load the trained KNN model and scaler"""
    try:
        model_path = 'models/knn_digit_classifier.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if not os.path.exists(model_path):
            return None, None, "Model file not found. Please train the model first."
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def preprocess_image(image_data):
    """Preprocess the drawn image to match MNIST format"""
    # Convert to grayscale if needed
    if len(image_data.shape) == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    
    # Invert colors (MNIST has white digits on black background)
    image_data = 255 - image_data
    
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
        new_w = int(20 * w / h)
    else:
        new_w = 20
        new_h = int(20 * h / w)
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 image with digit centered
    final_image = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize
    final_image = final_image.astype(np.float32)
    
    return final_image

# Load model
model, scaler, error = load_model()

if error:
    st.error(error)
    st.info("👉 Please run the Jupyter notebook to train the model first.")
    st.code("jupyter notebook notebooks/knn_digit_classifier.ipynb")
    st.stop()

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🎲 Try Sample", "ℹ️ How to Use"])

with tab1:
    st.subheader("Upload Your Handwritten Digit")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a handwritten digit"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            if st.button("🎯 Predict", type="primary", use_container_width=True):
                with st.spinner("🤔 Analyzing your digit..."):
                    # Preprocess image
                    processed_image = preprocess_image(image_array)
                    
                    if processed_image is None:
                        st.error("Could not process the image. Please try another image.")
                    else:
                        # Store in session state for display in col2
                        st.session_state['processed_image'] = processed_image
                        st.session_state['prediction_ready'] = True
    
    with col2:
        if 'prediction_ready' in st.session_state and st.session_state['prediction_ready']:
            processed_image = st.session_state['processed_image']
            
            # Display preprocessed image
            st.write("**Preprocessed Image (28x28):**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(processed_image, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
            # Flatten and scale
            image_vector = processed_image.flatten().reshape(1, -1)
            image_scaled = scaler.transform(image_vector)
            
            # Predict
            prediction = model.predict(image_scaled)[0]
            probabilities = model.predict_proba(image_scaled)[0]
            confidence = probabilities[prediction] * 100
            
            # Display prediction
            st.markdown(f'<div class="prediction-box">Predicted Digit: {prediction}</div>', 
                      unsafe_allow_html=True)
            
            st.markdown(f'<div class="confidence-box"><b>Confidence:</b> {confidence:.1f}%</div>', 
                      unsafe_allow_html=True)
            
            # Show top 3 predictions
            st.write("**Top 3 Predictions:**")
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            
            for idx in top_3_indices:
                prob = probabilities[idx] * 100
                st.progress(prob / 100)
                st.write(f"Digit **{idx}**: {prob:.1f}%")
            
            # Show neighbor information
            st.write("---")
            st.write(f"**Algorithm:** K-Nearest Neighbors")
            st.write(f"**K value:** {model.n_neighbors}")
            
            # Get distances and indices of neighbors
            distances, indices = model.kneighbors(image_scaled)
            st.write(f"**Average distance to neighbors:** {distances[0].mean():.2f}")
            
            # Clear button
            if st.button("🗑️ Clear Results"):
                st.session_state['prediction_ready'] = False
                st.rerun()
        else:
            st.info("👈 Upload an image and click **Predict** to see results")
            st.write("**Tips for best results:**")
            st.write("- Use a clear image of a single digit")
            st.write("- White or light background works best")
            st.write("- Center the digit in the image")
            st.write("- Avoid too much noise or clutter")

with tab2:
    st.subheader("Try with Sample MNIST Digits")
    
    # Load sample MNIST digits
    try:
        from sklearn.datasets import fetch_openml
        
        if 'sample_digits' not in st.session_state:
            with st.spinner("Loading sample digits..."):
                mnist = fetch_openml('mnist_784', version=1, parser='auto')
                # Get 10 random samples
                indices = np.random.choice(len(mnist.data), 10, replace=False)
                st.session_state['sample_digits'] = mnist.data.iloc[indices].values
                st.session_state['sample_labels'] = mnist.target.iloc[indices].values.astype(int)
        
        # Display samples in a grid
        cols = st.columns(5)
        for i in range(10):
            col_idx = i % 5
            with cols[col_idx]:
                digit_image = st.session_state['sample_digits'][i].reshape(28, 28)
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(digit_image, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                
                if st.button(f"Test #{i+1}", key=f"sample_{i}", use_container_width=True):
                    # Predict
                    image_scaled = scaler.transform(st.session_state['sample_digits'][i].reshape(1, -1))
                    prediction = model.predict(image_scaled)[0]
                    true_label = st.session_state['sample_labels'][i]
                    
                    if prediction == true_label:
                        st.success(f"✅ Correct! Predicted: {prediction}")
                    else:
                        st.error(f"❌ Wrong! Predicted: {prediction}, Actual: {true_label}")
    
    except Exception as e:
        st.error(f"Could not load sample digits: {str(e)}")
        st.info("Sample digits require internet connection on first run.")

with tab3:
    st.subheader("How to Use This Application")
    
    st.write("""
    ### 📤 Upload Image Tab
    1. Click on "Browse files" or drag and drop an image
    2. The image should contain a single handwritten digit (0-9)
    3. Click the "Predict" button to classify the digit
    4. View the results including confidence scores
    
    ### 🎲 Try Sample Tab
    - Test the model with real MNIST digits
    - Click any "Test" button to see predictions
    - Green checkmark means correct prediction
    - Red X means incorrect prediction
    
    ### 💡 Tips for Best Results
    - Use images with clear, centered digits
    - Light background with dark digit works best
    - Avoid images with multiple digits
    - Remove excessive noise or clutter
    
    ### 🎨 Creating Your Own Test Images
    You can create test images using:
    - Paint/Drawing applications
    - Tablet/stylus for natural handwriting
    - Smartphone apps
    - Screenshots of handwritten digits
    
    ### 📊 Understanding the Results
    - **Predicted Digit**: The model's best guess
    - **Confidence**: How sure the model is (0-100%)
    - **Top 3 Predictions**: Alternative predictions with probabilities
    - **K value**: Number of neighbors used in KNN algorithm
    """)

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.metric("Algorithm", "KNN")
with col_f2:
    st.metric("Dataset", "MNIST")
with col_f3:
    st.metric("Classes", "10 digits")

st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Streamlit and scikit-learn</p>
    <p>📚 Educational project for teaching machine learning</p>
</div>
""", unsafe_allow_html=True)
