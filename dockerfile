FROM continuumio/miniconda3:4.12.0 AS compile-image


WORKDIR /workspace/

# COPY ./vits /opt/conda/envs/vits
# COPY ./vits_web_demo/ /workspace/

# Make RUN commands use the new environment:
RUN echo "conda activate vits" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]


