#!/bin/bash

# TrustOps Virtual Environment Setup Script

set -e

echo "========================================="
echo "TrustOps Virtual Environment Setup"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

echo "Detected Python version: ${PYTHON_VERSION}"

# Simple version check
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "Error: Python 3.11 or higher is required"
    exit 1
fi

echo "✓ Python version check passed"

# Create virtual environment
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf venv
fi

echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Install TrustOps CLI in development mode
echo "Installing TrustOps CLI..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ TrustOps CLI installed (run 'trustops --help' to verify)"
else
    echo "✗ Failed to install TrustOps CLI"
    exit 1
fi

# Copy env.example to .env if it doesn't exist
if [ ! -f .env ] && [ -f infrastructure/env.example ]; then
    cp infrastructure/env.example .env
    echo "✓ Created .env from infrastructure/env.example"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
