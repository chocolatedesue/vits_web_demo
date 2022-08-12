sudo apt update && sudo apt upgrade -y 
sudo apt-get install -y  libsndfile1-dev tmux build-essential cmake libsndfile1  espeak python

git clone https://github.com/chocolatedesue/vits_web_demo.git
cd vits

cd monotonic_align
python setup.py build_ext --inplace
mv build/lib.linux-x86_64-3.8/monotonic_align ./ -r
python setup.py build_ext --inplace
cd ..

wget https://link.jscdn.cn/sharepoint/aHR0cHM6Ly9zdHV4aWRpYW5lZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC8yMTAwOTIwMDQzMV9zdHVfeGlkaWFuX2VkdV9jbi9FVTkybE9JVnY2NURtaHRwZHdCT0w0MEJHYktySG04ZHVaV0VCRkpSb0VmSnFnP2U9YUlnWVlk.jpg -O dalao_model -P asset/model 

wget https://link.jscdn.cn/sharepoint/aHR0cHM6Ly9zdHV4aWRpYW5lZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC8yMTAwOTIwMDQzMV9zdHVfeGlkaWFuX2VkdV9jbi9FUUNUS0EtVUFSVkx0eUtDa3laYTFUTUJNSGxIeTFPdjNuaFRYNExmbHRQNmNnP2U9bDU3NFVB.jpg -O dalao_config -P asset/config 


# python app.py