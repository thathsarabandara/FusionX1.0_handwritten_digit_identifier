"""
Script to train the KNN digit classifier model
Run this script to train and save the model before using the web app
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import time
import argparse


def load_mnist_data(subset_size=None):
    """
    Load MNIST dataset
    
    Args:
        subset_size: If specified, use only this many samples (for faster training)
        
    Returns:
        X, y: Features and labels
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).astype(int)
    
    if subset_size and subset_size < len(X):
        print(f"Using subset of {subset_size} samples...")
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_knn_model(X_train, y_train, k=3):
    """
    Train KNN classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        k: Number of neighbors
        
    Returns:
        Trained KNN model
    """
    print(f"\nTraining KNN model with K={k}...")
    start_time = time.time()
    
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    return knn


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        accuracy: Test accuracy
    """
    print("\nEvaluating model...")
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    
    pred_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Prediction completed in {pred_time:.2f} seconds")
    print(f"Average time per sample: {pred_time/len(X_test)*1000:.2f} ms")
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Detailed classification report
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
    return accuracy


def find_best_k(X_train, y_train, X_val, y_val, k_values=[1, 3, 5, 7, 9]):
    """
    Find the best K value using validation set
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        k_values: List of K values to try
        
    Returns:
        best_k: Best K value
    """
    print("\nFinding best K value...")
    best_k = k_values[0]
    best_accuracy = 0
    
    for k in k_values:
        print(f"Testing K={k}...", end=' ')
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy: {accuracy*100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    print(f"\nBest K: {best_k} with accuracy: {best_accuracy*100:.2f}%")
    return best_k


def save_model(model, scaler, output_dir='models'):
    """
    Save trained model and scaler
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'knn_digit_classifier.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✅ Model saved successfully!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train KNN digit classifier')
    parser.add_argument('--subset-size', type=int, default=10000,
                        help='Number of samples to use (default: 10000, use 0 for full dataset)')
    parser.add_argument('--k', type=int, default=None,
                        help='K value for KNN (default: auto-tune)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 60)
    print("KNN Handwritten Digit Classifier - Training Script")
    print("=" * 60)
    
    # Load data
    subset_size = args.subset_size if args.subset_size > 0 else None
    X, y = load_mnist_data(subset_size)
    
    # Split data
    print(f"\nSplitting data (test size: {args.test_size*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling completed")
    
    # Find best K or use specified K
    if args.k is None:
        # Split training set for validation
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        best_k = find_best_k(X_train_sub, y_train_sub, X_val, y_val)
    else:
        best_k = args.k
        print(f"\nUsing specified K={best_k}")
    
    # Train final model
    final_model = train_knn_model(X_train_scaled, y_train, k=best_k)
    
    # Evaluate
    accuracy = evaluate_model(final_model, X_test_scaled, y_test)
    
    # Save model
    save_model(final_model, scaler, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final Model: K={best_k}, Accuracy={accuracy*100:.2f}%")
    print("=" * 60)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
