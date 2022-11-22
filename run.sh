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