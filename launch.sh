#!/bin/bash

# create the Python virtual env and install the requirements
cd ragnarok
python3 -m venv venv
source venv/bin/activate
# default llama-cpp-python does not support cuda, have to provide cmake_args.
export LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python==0.2.26

# You can change the visible device values here to specify what GPU(s) that you want RAGnarok to use (or not use). 
# For this example I'm only allowing RAGnarok to use GPU device 0.
# See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars for more details. 
export CUDA_VISIBLE_DEVICES=0

pip3 install -r requirements.txt

# kick off the main app
streamlit run RAGnarok_Settings.py
