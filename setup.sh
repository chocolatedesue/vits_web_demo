# sudo su 
conda create -n dl  python=3.9 -y
conda init bash
bash

conda activate dl
export POETRY_VERSION=1.3.1
export DEBIAN_FRONTEND=noninteractive && \
    sudo apt-get update && \
    sudo apt-get install cmake build-essential  -y --no-install-recommends && \
    pip install poetry==$POETRY_VERSION 


poetry export -f requirements.txt -o requirements.txt --without dev --without test  --without-hashes  && \
pip install --upgrade pip  && \
pip install -r requirements.txt  



function run{
    python -m app.main
}