
# git clone https://github.com/chocolatedesue/vits_web_demo.git
# cd vits

sudo DEBIAN_FRONTEND=noninteractive apt update &&  sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y 
sudo  DEBIAN_FRONTEND=noninteractive apt-get install -y  libsndfile1-dev tmux build-essential cmake libsndfile1  espeak python3-pip ffmpeg
pip install -r requirements.txt

cd monotonic_align
python3 setup.py build_ext --inplace
cd ..

wget https://link.jscdn.cn/sharepoint/aHR0cHM6Ly9zdHV4aWRpYW5lZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC8yMTAwOTIwMDQzMV9zdHVfeGlkaWFuX2VkdV9jbi9FVTkybE9JVnY2NURtaHRwZHdCT0w0MEJHYktySG04ZHVaV0VCRkpSb0VmSnFnP2U9YUlnWVlk.jpg -O asset/model/dalao_model.pth  

wget https://link.jscdn.cn/sharepoint/aHR0cHM6Ly9zdHV4aWRpYW5lZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC8yMTAwOTIwMDQzMV9zdHVfeGlkaWFuX2VkdV9jbi9FUUNUS0EtVUFSVkx0eUtDa3laYTFUTUJNSGxIeTFPdjNuaFRYNExmbHRQNmNnP2U9bDU3NFVB.jpg -O asset/configs/dalao_config.json


# python3 app.py