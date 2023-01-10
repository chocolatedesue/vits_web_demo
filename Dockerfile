
# FROM python:3.11.1-slim-bullseye as compile-image
FROM python:3.9.15-slim-bullseye as compile-image
ENV POETRY_VERSION=1.3.1


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install cmake build-essential libsndfile-dev -y --no-install-recommends && \
    pip install poetry==$POETRY_VERSION  

COPY ./pyproject.toml ./poetry.lock ./


RUN poetry export -f requirements.txt -o requirements.txt --without dev --without test  --without-hashes  && \
    python -m venv /opt/venv && \
    # /opt/venv/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt  && \
    /opt/venv/bin/pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu 


COPY ./app ./app

RUN  cd /app && \
    cd monotonic_align && \
    /opt/venv/bin/python setup.py build_ext --inplace 

# FROM python:3.11.1-slim-bullseye as final
FROM python:3.9.15-slim-bullseye as final
EXPOSE 7860
COPY --from=compile-image /opt/venv /opt/venv
COPY --from=compile-image /app /app
ENV TZ=Asia/Shanghai PATH="/opt/venv/bin:$PATH"


WORKDIR /app
RUN python init_jptalk.py  && \
    apt-get update && \
    apt-get install -y --no-install-recommends libsndfile-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /root/.cache/pypoetry    



CMD ["python", "main.py"]