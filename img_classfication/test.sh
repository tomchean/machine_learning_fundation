#!/bin/bash
wget https://github.com/tomchean/Search-Engine/releases/download/v1.0/best_model
python3 predict.py $1 $2
