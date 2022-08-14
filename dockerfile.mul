FROM ubuntu:20.04 as compile

COPY . /workspace/
WORKDIR /workspace/

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y  build-essential cmake python3-pip  && \
    pip3 wheel  --wheel-dir=/wheel   -r requirements.txt && \
    python3 -m pip install --no-index --find-links=/wheel -r /workspace/requirements.txt && \
    cd monotonic_align && \
    python3 setup.py build_ext --inplace 



FROM ubuntu:20.04

WORKDIR /workspace/
COPY --from=compile /wheel /wheel
COPY --from=compile /workspace/ /workspace/

RUN export debian_frontend=noninteractive  && \ 
    apt update && \ 
    apt-get -y --no-install-recommends install python3-pip libsndfile1-dev && \
    apt clean && \ 
    python3 -m pip install --no-index --find-links=/wheel -r /workspace/requirements.txt && \
    python3 init_jptalk.py && \
    rm -rf /wheel 



