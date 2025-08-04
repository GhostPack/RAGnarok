#!/bin/bash

# create the Python virtual env and install the requirements
cd ragnarok
sudo apt install python3.11-venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# kick off the main app
streamlit run RAGnarok_Settings.py
