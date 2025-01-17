#!/bin/bash

# create the Python virtual env and install the requirements
cd ragnarok
python3 -m venv venv
source venv/bin/activate
export LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python==0.2.26
export CUDA_VISIBLE_DEVICES=0,1
pip3 install -r requirements.txt

# kick off the main app
streamlit run RAGnarok_Settings.py
