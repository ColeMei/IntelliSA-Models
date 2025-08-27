#!/bin/bash
# Local virtual environment setup

echo "Setting up local virtual environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Local environment setup complete!"
echo "Activate with: source environments/venv/bin/activate"
