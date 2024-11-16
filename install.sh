#!/bin/bash

# Update pip to the latest version
pip install --upgrade pip

# Install core libraries
pip install dropbox numpy scipy pandas matplotlib tqdm ipywidgets h5py

# Install medical imaging libraries
pip install SimpleITK nibabel

# Install PyTorch (specific version may vary based on your system)
pip install torch torchvision torchaudio

# Install scikit-image
pip install scikit-image

# Install image quality assessment library (piq)
pip install piq

# Suppress warnings
pip install warnings
