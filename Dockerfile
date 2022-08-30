FROM continuumio/miniconda3:4.12.0 


WORKDIR /workspace/
COPY . /workspace/ 

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y cmake build-essential libsndfile1-dev && \
    conda env create -f conda_env.yml  && \
    echo "conda activate vits" >> ~/.bashrc && \
    apt-get remove  -y cmake build-essential && \
    apt-get clean && \
    apt autoremove -y && \
    conda clean -p   


    
SHELL ["/bin/bash", "--login", "-c"]

RUN python init_jptalk.py && \
    cd monotonic_align && \
    python setup.py build_ext --inplace && \
    pip cache remove purge  
