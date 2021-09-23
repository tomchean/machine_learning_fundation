#!/bin/bash
wget https://github.com/tomchean/Search-Engine/releases/download/v1.0/best_model
wget https://github.com/tomchean/Search-Engine/releases/download/hw5/model_20
wget https://github.com/tomchean/Search-Engine/releases/download/hw5/model_10
python3 produce.py $1 $2
