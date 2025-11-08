# Handwritten Digit Classifier using KNN

An educational project demonstrating handwritten digit classification using K-Nearest Neighbors (KNN) algorithm on the MNIST dataset.

## 🎯 Project Overview

This repository contains:
- **Educational Jupyter Notebook**: Step-by-step tutorial on KNN classification
- **Interactive Demo**: Streamlit web application for real-time digit recognition
- **Training Scripts**: Code to train and save KNN models

## 📚 What You'll Learn

- Understanding the KNN algorithm
- Working with the MNIST dataset
- Feature extraction and preprocessing
- Model evaluation and optimization
- Deploying ML models as web applications

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/thathsarabandara/FusionX1.0_handwritten_digit_identifier.git
cd FusionX1.0_handwritten_digit_identifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Jupyter Notebook

```bash
jupyter notebook notebooks/knn_digit_classifier.ipynb
```

### Running the Demo Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
FusionX1.0_handwritten_digit_identifier/
├── app.py                          # Streamlit demo application
├── notebooks/
│   └── knn_digit_classifier.ipynb  # Educational notebook
├── src/
│   ├── train_model.py              # Model training script
│   └── utils.py                    # Helper functions
├── models/                         # Saved trained models
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎓 Educational Content

### KNN Algorithm Basics

K-Nearest Neighbors is a simple, intuitive algorithm that:
1. Stores all training examples
2. For a new input, finds the K closest training examples
3. Assigns the most common class among those K neighbors

### Why KNN for Digit Recognition?

- **Simple to understand**: Great for teaching ML concepts
- **No training phase**: Model stores the data directly
- **Interpretable**: You can visualize why a prediction was made
- **Effective**: Achieves good accuracy on MNIST

## 📊 Expected Performance

- **Accuracy**: ~97% on MNIST test set (with K=3)
- **Inference Time**: ~0.1-0.5 seconds per digit
- **Model Size**: ~50MB (for subset of training data)

## 🛠️ Technologies Used

- **scikit-learn**: KNN implementation
- **NumPy**: Numerical computations
- **Matplotlib**: Visualizations
- **Streamlit**: Web application framework
- **Pillow**: Image processing
- **OpenCV**: Image preprocessing

## 📝 License

MIT License - Feel free to use this for educational purposes!

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🎉**
