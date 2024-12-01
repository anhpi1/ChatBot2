ALTER AUTHORIZATION ON DATABASE::comparator TO sa;

vào sql server restore data base in model\data\comparator.bak
vào model\data\tham_so.py chỉnh sửa kết nối cho sql bao gồm
ví dụ:
server = 'DESKTOP-1MU0IU3\SQLEXPRESS'
database = 'comparator'
username = ''
password = ''

run 
model\lay_du_lieu_tu_sql.py
model\lay_du_lieu_train_tu_excel.py
model\tao_word_list.py
model\train.py
model\du_doan.py

note: report không bao gồm học tăng cường

cách dùng api
url: http://127.0.0.1:5000/
vd dữ liệu gửi cho api
{
  "message": "what is comparator"
  "true_label": [0, 24, 12, 0, 0, 0, 0, 0, 0] (bắt buộc có nếu dùng du_doan_co_lich_su)
}

vd api gửi về
{
  "message": "Comparator is comparator"
}
hoặc
{
  "message": "error"        (nếu có lỗi)
}
