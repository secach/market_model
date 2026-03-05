#!/bin/bash

# go to your project directory
cd ~/Projects/market_model   # replace with your actual path on Mac

# activate your virtual environment
source .venv/bin/activate

# run the Python program
python3 run.py

# wait for user input before closing
read -p "Press enter to exit"