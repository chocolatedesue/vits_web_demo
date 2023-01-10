## Copy the demo web from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts/tree/main) 
> Thanks a lot to [@CjangCjengh](https://github.com/CjangCjengh)
> only support  japanese_cleaners and japanese_cleaners2 in config.json with raw vits model

***Only used for entertainment.
Don't used for bussiness***

## usage 

### one-click example
```shell
# use default model and config
docker run -d \
--name demo \
-p 7860:7860   \
ccdesue/vits_demo
```


### From docker 
 1. sudo apt update && sudo apt install -y  docker.io
 2. download the model and config with the postfix .json and .pth
 3. run the cmd (change -v param to tail to your model and config dir)  
 
Example:
```docker 
docker run -d \
--name demo \
-p 7860:7860   \
-v  /path/to/model_dir:/app/.model \
ccdesue/vits_demo    
```


> You can alos build your own docker images or server with the repo's code 

## Helpful info
1. please read the source code to better understand
2. refer to the demo config.json to tail to your own model config
3. refer the dockerfile 




## license 
GPLv2
