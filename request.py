import requests

url = "http://localhost:8888/chat"
response = requests.post(url, json={"prompt": "Bạn có thể giúp tôi viết bài báo không?"})

print(response.json())  # Kết quả từ DeepSeek
