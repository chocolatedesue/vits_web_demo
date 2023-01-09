docker build -t ccdesue/vits_demo:onnx .

function run()
{
export name=vits_onnx
docker stop $name
docker rm $name
docker run -d \
--name $name \
-p 7860:7860 \
ccdesue/vits_demo:onnx
}

# docker run --rm -it -p 7860:7860/tcp ccdesue/vits_demo:onnx  bash

function push(){
    docker push ccdesue/vits_demo:onnx
}
