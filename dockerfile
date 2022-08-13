FROM ubuntu:20.04

WORKDIR /workspace/
COPY . /workspace/



RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y  libsndfile1-dev cmake libsndfile1  espeak python3-pip  && \
    python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt && \
    python3 init_jptalk.py && \
    cd monotonic_align && \
    python3 setup.py build_ext --inplace


CMD ["python3","app.py","-m /mydata/model.pth","-c /mydata/config.json"]