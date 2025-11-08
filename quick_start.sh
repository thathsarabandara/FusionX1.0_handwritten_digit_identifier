#!/bin/bash

# Quick Start Script for Handwritten Digit Classifier
# This script sets up the environment and trains the model

echo "=========================================="
echo "Handwritten Digit Classifier - Quick Start"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed successfully!"
echo ""

# Ask user if they want to open the notebook
read -p "Do you want to open the Jupyter notebook for learning? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📓 Setting up Jupyter kernel..."
    
    # Install ipykernel if not already installed
    pip install ipykernel -q
    
    # Create kernel for this project
    python -m ipykernel install --user --name=handwritten_digit_classifier --display-name "Python (Digit Classifier)" > /dev/null 2>&1
    
    echo "✅ Kernel 'Python (Digit Classifier)' created successfully!"
    echo ""
    echo "🚀 Opening Jupyter notebook..."
    echo "   Please select 'Python (Digit Classifier)' as the kernel in Jupyter"
    echo ""
    
    # Open Jupyter notebook
    jupyter notebook notebooks/knn_digit_classifier.ipynb
    
else
    echo ""
    # Ask user if they want to train the model
    read -p "Do you want to train the model now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🎓 Training KNN model..."
        echo "This will use 10,000 samples and may take a few minutes..."
        echo ""
        
        python src/train_model.py --subset-size 10000
        
        echo ""
        echo "✅ Model training completed!"
    else
        echo ""
        echo "⚠️  Skipping model training."
        echo "You can train the model later by running:"
        echo "  python src/train_model.py --subset-size 10000"
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""

# Ask if user wants to run the app locally
read -p "Do you want to run the Streamlit app now? (y/n): " run_app

if [[ $run_app =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting Streamlit app..."
    echo ""
    
    # Get local IP address
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        LOCAL_IP=$(hostname -I | awk '{print $1}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        LOCAL_IP=$(ipconfig getifaddr en0)
    else
        LOCAL_IP="localhost"
    fi
    
    # Generate QR code for mobile access
    echo "📱 Scan this QR code to access from your mobile device:"
    echo ""
    
    # Check if qrencode is available
    if command -v qrencode &> /dev/null; then
        echo "http://${LOCAL_IP}:8501" | qrencode -t ANSIUTF8
    else
        echo "⚠️  QR code generator not found. Install it with:"
        echo "   Ubuntu/Debian: sudo apt-get install qrencode"
        echo "   macOS: brew install qrencode"
        echo ""
        echo "Or access directly at: http://${LOCAL_IP}:8501"
    fi
    
    echo ""
    echo "🌐 Access URLs:"
    echo "   Local:    http://localhost:8501"
    echo "   Network:  http://${LOCAL_IP}:8501"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Run Streamlit and open browser
    streamlit run app.py --server.headless true --browser.gatherUsageStats false
else
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Run the Jupyter notebook for learning:"
    echo "   jupyter notebook notebooks/knn_digit_classifier.ipynb"
    echo ""
    echo "2. Run the Streamlit demo app:"
    echo "   streamlit run app.py"
    echo ""
    echo "3. Train a custom model:"
    echo "   python src/train_model.py --help"
    echo ""
    echo "For deployment instructions, see DEPLOYMENT.md"
    echo ""
fi
