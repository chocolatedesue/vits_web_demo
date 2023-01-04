FROM python:3.9.14-slim as build 
ENV POETRY_VERSION=1.3.1
RUN pip install poetry==$POETRY_VERSION
COPY ./pyproject.toml ./poetry.lock ./
RUN poetry export -f requirements.txt -o requirements.txt  --without-hashes

FROM python:3.9.14-slim as compile-image


COPY --from=build requirements.txt .
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install cmake build-essential libsndfile-dev -y --no-install-recommends && \    
    python -m pip install --upgrade pip && \
    pip install --user -r requirements.txt --no-cache-dir 

COPY ./app/ /app

RUN cd /app && \
    cd monotonic_align && \
    python setup.py build_ext --inplace 


FROM python:3.9.14-slim as final
EXPOSE 7860
COPY --from=compile-image /root/.local /root/.local 
COPY --from=compile-image /app /app
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
RUN python init_jptalk.py  && \
    apt-get update && \
    apt-get install -y --no-install-recommends libsndfile-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip && \
    rm -rf /root/.cache/pypoetry    



CMD ["python", "app.py"]