#!/bin/bash
wget https://github.com/tomchean/Search-Engine/releases/download/rnn/model.tar.gz
tar -zxvf model.tar.gz
python3 predict.py $1 $2
