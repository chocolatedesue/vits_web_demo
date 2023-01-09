## onnx inference server in docker container

### Copy the demo web from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts/tree/main) 
> Thanks a lot to [@CjangCjengh](https://github.com/CjangCjengh)


***Only used for entertainment.
Don't used for bussiness***

### quick start 
```shell
export name=vits_onnx
docker stop $name
docker rm $name
docker run -d \
--name $name \
-p 7860:7860 \
ccdesue/vits_demo:onnx
```


### dir structure
```

├── app             # gradio code 
├── build.sh
├── Dockerfile      
├── export          # some util for export model
├── LICENSE
├── poetry.lock
├── __pycache__
├── pyproject.toml
├── README.md
├── setup.sh
└── util        # some posibile util 

```

### Helpful info
1. please read the source code to better understand
2. refer to the demo config.json to tail to your own model config
3. refer the dockerfile 

### limitation
1. unsupported to adjust infer params at running time. It is fixed on model export stage
2. only  test  on japanese_cleaners and japanese_cleaners2 in config.json with  [raw vits](https://github.com/jaywalnut310/vits)


### Reference
1. [vits_export_discussion](https://github.com/MasayaKawamura/MB-iSTFT-VITS/issues/8)
2. [other_vits_onnx](https://github.com/NaruseMioShirakana/VitsOnnx)
3. [wetts](https://github.com/wenet-e2e/wetts)


### license 
GPLv2