#!/bin/bash
wget https://github.com/tomchean/Search-Engine/releases/download/seq/seq2seq.ckpt
python3 test.py $1 $2
