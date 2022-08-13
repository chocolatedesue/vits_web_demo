docker pull 

wget $model_path -O ~/.model/model.pth
wget $config_path -O ~/.model/config.json

docker run -d \
--name demo \
-p 7860:7860   \
-v  ~/.moedl/:/mydata \
vits_demo