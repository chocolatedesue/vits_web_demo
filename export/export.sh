
function get_data(){
    mkdir -p model 
    cd model 
    model_url='https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJiTzdqanlEQXNyWDV4bDA/root/content'
    config_url='https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdG53cTVRejJnLTJhNEJ3enhhUHpqNE5EZWc/root/content'
    
    wget -O model.pth $model_url
    wget -O  config.json $config_url
    cd ..
}


# mkdir -p model

python vits/export_onnx.py --checkpoint model/model.pth --cfg model/config.json \
    --onnx_model model/model.onnx 

# https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBb1E3R21YN3hRWkpnYjQtT1VocVdjUFc4VWM5bVE/root/content