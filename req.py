import requests
import json

url = "http://localhost:26302/generate"
response = requests.post(url, json={"topic": "未来守护者杰斯"}, stream=True)

for chunk in response.iter_lines():
    if chunk:
        print(chunk.decode('utf-8'))
        # data = json.loads(chunk.decode('utf-8'))
        # print(data)
        print("========================================================")
