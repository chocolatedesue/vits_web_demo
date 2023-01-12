python -m pip install --upgrade pip && \
export POETRY_VERSION=1.3.1
pip install poetry==$POETRY_VERSION

export DEBIAN_FRONTEND=noninteractive && \
sudo apt-get update && \
sudo apt-get install cmake build-essential libsndfile-dev -y --no-install-recommends && \
poetry export -f requirements.txt -o requirements.txt --without-hashes && \
pip install  -r requirements.txt

pip3 install torch==1.13  --extra-index-url https://download.pytorch.org/whl/cpu
#pip install torch-directml

cd app/monotonic_align && \
python setup.py build_ext --inplace && \
cd ..

function build_docker() {
    docker build -t ccdesue/vits_demo .
}
