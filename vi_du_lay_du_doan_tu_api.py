import requests

# URL của API
url = 'http://127.0.0.1:5000/du_doan'

# Chuỗi ký tự muốn gửi
data = {
    "message": "what is comparator"
}

# Gửi yêu cầu POST với chuỗi ký tự
response = requests.post(url, json=data)

# In kết quả trả về từ API
print(response.json())
