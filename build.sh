docker build -t ccdesue/vits_demo:onnx .

function run()
{
# export name=vits_onnx
docker stop vits_onnx
docker rm vits_onnx
docker run -d \
--name vits_onnx \
-p 7860:7860 \
ccdesue/vits_demo:onnx
}

function push(){
    docker push ccdesue/vits_demo:onnx
}
