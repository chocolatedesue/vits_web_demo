FROM mid AS compile-image

FROM ubuntu:20.04
WORKDIR /workspace/

COPY --from=compile-image /opt/venv /opt/venv 
COPY --from=compile-image /workspace/ /workspace/
ENV PATH="/opt/venv/bin:$PATH"

# docker run -itd \
# --name demo \
# -p 7860:7860   \
# -v  ~/.model:/mydata \
# -e PATH=/root/.local/bin:$PATH \
# fs  /bin/bash 

# python3 app.py -m /mydata/model.pth -c /mydata/config.json

# python3 /workspace/app.py -m /mydata/model.pth -c /mydata/config.json