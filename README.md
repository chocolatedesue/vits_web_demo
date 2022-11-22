## Copy the demo web from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts/tree/main) 
> Thanks a lot to [@CjangCjengh](https://github.com/CjangCjengh)

***Only used for self-entertainment.
Don't used for bussiness***

## usage 

## one-click example
```shell
model_url='https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJiTzdqanlEQXNyWDV4bDA/root/content'
config_url='https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content'

mkdir -p ~/model
wget -P ~/model -O  model.pth $model_url
wget -P ~/model -O config.json $config_url
docker run -d \
--name demo \
-p 7860:7860   \
-v  ~/model:/mydata \
ccdesue/vits_demo
```


### From docker 
 1. sudo apt update && sudo apt install -y  docker.io
 2. download the model and config with the postfix .json and .pth
 3. run the cmd (change -v param to tail to your model and config dir)
```docker 
docker run -d \
--name demo \
-p 7860:7860   \
-v  /mydata:/mydata \
ccdesue/vits_demo   
```


### From source
1. sudo apt install  cmake build-essential libsndfile1-dev python3-pip python-is-python3 -y
2. pip install torch==1.11.0+cpu  --extra-index-url https://download.pytorch.org/whl/cpu
3. pip install -r requirements.txt
3. python app.py

> You can alos build your own docker images or server with the repo's code 

## Helpful info
1. please read the source code to better understand
2. refer to the demo config.json to tail to your own model config

## pretrained model and config
yuzusoft*3 + ambitious mission (direct link)
1. [model](https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJiTzdqanlEQXNyWDV4bDA/root/content)
2. [config](https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content)  




## license 
GPLv2
