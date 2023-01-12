# create time : 2023.1.09
# onnxruntime==1.13.1 not supported py=3.11
# FROM python:3.11.1-slim-bullseye as compile-image
FROM python:3.9.16-slim-bullseye as compile-image
# FROM  python:3.9.16-alpine as compile-image
ENV POETRY_VERSION=1.3.1

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install cmake build-essential  -y --no-install-recommends && \
    pip install --no-cache-dir poetry==$POETRY_VERSION


COPY ./pyproject.toml ./app/init_jptalk.py ./poetry.lock ./
RUN  poetry export -f requirements.txt -o requirements.txt --without dev --without test  --without-hashes  && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt   && \
    /opt/venv/bin/python3  init_jptalk.py 

# FROM python:3.11.1-slim-bullseye as final
FROM python:3.9.16-slim-bullseye as final
# FROM  python:3.9.16-alpine as compile-final
EXPOSE 7860
COPY --from=compile-image /opt/venv /opt/venv
# COPY ./app/init_jptalk.py /app/init_jptalk.py
ENV TZ=Asia/Shanghai PATH="/opt/venv/bin:$PATH"
COPY gr/app /app
WORKDIR /
# use for huggingface
RUN mkdir -p /app/.model && \
    chmod -R  777  /app 
CMD ["python", "-m","app.main"]