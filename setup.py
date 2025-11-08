"""
Setup script for handwritten digit classifier
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="handwritten-digit-classifier",
    version="1.0.0",
    author="Thathsara Bandara",
    description="KNN-based handwritten digit classifier for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thathsarabandara/FusionX1.0_handwritten_digit_identifier.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "streamlit>=1.28.0",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0.76",
        "joblib>=1.3.2",
        "seaborn>=0.12.2",
    ],
)
