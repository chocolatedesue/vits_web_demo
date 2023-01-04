import requests

response = requests.post("http://127.0.0.1:7860/run/tts", json={
  "data": [
    "こんにちは、あやち寧々です。",
    "0:nen",
    1,
]}).json()

data = response["data"]
print(data)