# update pip 

function setup_py(){
conda create -n dl python=3.9 -y
conda init bash
bash
}

conda activate dl 
export DEBIAN_FRONTEND=noninteractive && \
    sudo apt-get update && \
    sudo apt-get install cmake build-essential  -y --no-install-recommends


pip install --upgrade pip
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install onnxruntime Cython
pip install -r requirements.txt
