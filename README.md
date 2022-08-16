## Copy the demo web from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts/tree/main) 
> Thanks a lot to [@CjangCjengh](https://github.com/CjangCjengh)

***Only used for self-entertainment.
Don't used for bussiness***

## usage 
### From docker 
 1. sudo apt update && sudo apt install -y  docker.io
 2. docker pull ccdesue/vits_demo
 3. download the model and config, which should be named as  ~/.model/model.pth, ~/.model/config.json
 4. enter the docker
```docker 
docker run -itd \
--name demo \
-p 7860:7860   \
-v  ~/.model:/mydata \
ccdesue/vits_demo   /bin/bash
```
5. attach the shell to docker 
6. python app.py -m /mydata/model.pth -c /mydata/config.json

### From source
1. git clone https://gitee.com/ccdesue/vits_web_demo.git
2. cd vits_web_demo && bash asset\script\start_in_server.sh
3. download model and config 
4. python3 app.py -m /path/to/model -c /path/to/config

> You can alos build your own docker images or server with the repo's code 

## Helpful info
1. please read the source code to better understand
2. refer to the demo config.json to tail to your own model config

## pretrained model and config
SABBAT OF THE WITCH + DRACU-RIOT!
1. [model](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/21009200431_stu_xidian_edu_cn/EX5T4-fzg1FLvI0JYQVXobEBnOgQR55iR9pl3LeySss5nw?e=DLOB1B)
2. [config](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/21009200431_stu_xidian_edu_cn/EXzGh5EAl5tBlYMBh9bZVjUBV6IRY8IJF9hlfUwOXsV0wA?e=Oo1Hh3)


## license 
GPLv2