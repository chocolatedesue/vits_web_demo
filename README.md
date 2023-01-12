## onnx inference server in docker container

> [huggingface demo](https://huggingface.co/spaces/ccds/19_onnx)

### Copy the demo web from [link](https://huggingface.co/spaces/skytnt/moe-japanese-tts/tree/main)

> Thanks a lot to [@CjangCjengh](https://github.com/CjangCjengh)
> Thanks a lot to [wetts](https://github.com/wenet-e2e/wetts)

***Only used for entertainment.
Don't used for bussiness***

### quick start

> To use other model and config<br> please use -v /path/to/dir:/app/.model to mount your model and config

```shell
export name=vits_onnx
docker stop $name
docker rm $name
docker run -d \
--name $name \
-p 7860:7860 \
ccdesue/vits_demo:onnx
# -v /path/to/dir:/app/.model
```

### Helpful info

1. please read the source code to better understand
2. refer to the demo config.json to tail to your own model config
3. refer the dockerfile
4. to use gpu, please install onnxruntime-gpu

### limitation

1. only test on japanese_cleaners and japanese_cleaners2 in config.json
   with  [raw vits](https://github.com/jaywalnut310/vits)

### Reference

1. [vits_export_discussion](https://github.com/MasayaKawamura/MB-iSTFT-VITS/issues/8)
2. [other_vits_onnx](https://github.com/NaruseMioShirakana/VitsOnnx)
3. [wetts](https://github.com/wenet-e2e/wetts)
4. [android_vits](https://github.com/weirdseed/Vits-Android-ncnn)

### license

GPLv2