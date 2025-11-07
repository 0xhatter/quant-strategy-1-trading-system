#!/bin/bash

echo "======================================================================"
echo "  Installing Dependencies for Quantitative Trading System"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

echo ""
echo "Installing required packages..."
echo ""

# Install packages
pip3 install pandas numpy matplotlib torch scikit-learn requests

echo ""
echo "======================================================================"
echo "  Installation Complete!"
echo "======================================================================"
echo ""
echo "Verifying installation..."
echo ""

# Verify installation
python3 -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sklearn
import requests

print('✓ pandas:', pd.__version__)
print('✓ numpy:', np.__version__)
print('✓ matplotlib:', plt.matplotlib.__version__)
print('✓ torch:', torch.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ requests:', requests.__version__)
print('')
print('All dependencies installed successfully!')
print('')
print('Next step: Run the example workflow')
print('  python3 example_workflow.py')
"

echo ""
echo "======================================================================"
