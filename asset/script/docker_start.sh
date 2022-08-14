model_path=https://link.jscdn.cn/sharepoint/aHR0cHM6Ly9zdHV4aWRpYW5lZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC8yMTAwOTIwMDQzMV9zdHVfeGlkaWFuX2VkdV9jbi9FVTkybE9JVnY2NURtaHRwZHdCT0w0MEJHYktySG04ZHVaV0VCRkpSb0VmSnFnP2U9YUlnWVlk.jpg

config_path=https://link.jscdn.cn/sharepoint/aHR0cHM6Ly9zdHV4aWRpYW5lZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC8yMTAwOTIwMDQzMV9zdHVfeGlkaWFuX2VkdV9jbi9FUUNUS0EtVUFSVkx0eUtDa3laYTFUTUJNSGxIeTFPdjNuaFRYNExmbHRQNmNnP2U9bDU3NFVB.jpg 

sudo mkdir -p ~/.model/
sudo chmod 777 -R ~/.model/


wget ${model_path} -O ~/.model/model.pth
wget ${config_path} -O ~/.model/config.json

docker run -itd \
--name demo \
-p 7860:7860   \
-v  ~/.model:/mydata \
vits_demo  /bin/bash /workspace/run.sh


#python3 app.py -m /mydata/model.pth -c /mydata/config.json
#python3 app.py -m ~/.model/model.pth -c ~/.model/config.json