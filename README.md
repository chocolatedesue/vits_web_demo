# Http api for raw vits
> env: python3.9+conda or python3.9 detail can be found at dockerfile




***Only used for self-entertainment.Not commercial***

## reference 
1. [azure api of vits](https://github.com/fumiama/MoeGoe)
2. [moe-tts on huggingface](https://huggingface.co/spaces/skytnt/moe-tts)


## usage 
### From docker 
 1. sudo apt update && sudo apt install -y  docker.io
 2. docker pull ccdesue/vits_httpapi
 3. download the model and config with the postfix .json and .pth
 4. run the cmd (change -v param to tail to your model and config dir)
```docker 
docker run -d \
--name demo \
-p 8088:8088   \
-v  /mydata:/mydata \
ccdesue/vits_httpapi    
```


### From source
1. sudo apt install  cmake build-essential libsndfile1-dev python3-pip python-is-python3 -y
2. pip install torch==1.11.0+cpu  --extra-index-url https://download.pytorch.org/whl/cpu
3. pip install -r requirements.txt
3. uvicorn main:app --port 8088



## license 
GPLv2