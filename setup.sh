sudo su 

export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install cmake build-essential  -y --no-install-recommends && \
    pip install poetry==$POETRY_VERSION 

export POETRY_VERSION=1.3.1
pip install poetry==$POETRY_VERSION  && \
poetry export -f requirements.txt -o requirements.txt --without dev --without test  --without-hashes  && \
pip install --upgrade pip  && \
pip install -r requirements.txt  



