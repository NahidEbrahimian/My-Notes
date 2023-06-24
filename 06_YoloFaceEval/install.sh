#!bin/bash

python3.7 -m venv onnx_env_gpu

source onnx_env_gpu/bin/activate

# First you need to Upgrade pip
pip install --upgrade pip

# Installing some packages from a requirement file from partdp servers
pip install -r requirements.txt 
