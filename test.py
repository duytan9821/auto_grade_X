import requests

url = "http://127.0.0.1:8000/auto_grade/"
file_path = "test3.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.json())
