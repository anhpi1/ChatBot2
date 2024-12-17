import pyodbc
import json
import data.tham_so as ts

def sql():
    # Kết nối tới SQL Server
    connection = pyodbc.connect(
        f"DRIVER={{SQL Server}};SERVER={ts.server};DATABASE={ts.database};UID={ts.username};PWD={ts.password}"
    )
    cursor = connection.cursor()
    
    # Lấy danh sách các bảng
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    tables = [row.TABLE_NAME for row in cursor.fetchall()]

    # Ghi tên các bảng vào file tables.json
    with open('model\\data\\json\\tables.json', 'w', encoding='utf-8') as file:
        json.dump(tables, file, indent=4, ensure_ascii=False)

    all_data = []  # Danh sách chứa toàn bộ dữ liệu để ghi ra JSON
    datas=[]
    for table in tables:
        if(table== "sysdiagrams" or table== "question"):
            continue
        query = f"SELECT id FROM {table}"
        cursor.execute(query)
        id = [row[0] for row in cursor.fetchall()]
        query = f"SELECT content FROM {table}"  
        cursor.execute(query)
        content = [row[0] for row in cursor.fetchall()]
        data=[]
        with open('model\\data\\json\\data_table_{}.json'.format(table), 'w', encoding='utf-8') as data_file:
            json.dump(content, data_file, indent=4, ensure_ascii=False)
        for i,c in zip(id,content):
            data.append({
                "id": i,
                "content" : c
            })
        datas.append({
            "table":table,
            "data":data
        })
    with open('model\\data\\json\\data_content_lable.json', 'w', encoding='utf-8') as data_file:
        json.dump(datas, data_file, indent=4, ensure_ascii=False)
    cursor.execute("""
        select * from answer
    """)
    rows = cursor.fetchall()

    # Lấy tên cột từ cursor.description
    columns = [column[0] for column in cursor.description]

    # Chuyển đổi dữ liệu thành danh sách các từ điển (mỗi dòng dữ liệu là một dictionary)
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))

    # Đóng kết nối
    cursor.close()
    connection.close()

    # Tạo một đối tượng để ghi vào file JSON
    output = {
        "table_name": "answer",  # Tên bảng
        "data": data  # Dữ liệu
    }

    # Ghi dữ liệu ra file JSON
    with open("model\\data\\json\\answer.json", "w", encoding="utf-8") as json_file:
        json.dump(output, json_file, ensure_ascii=False, indent=4)
    # Đường dẫn tới tệp JSON
    file_path = 'model\\data\\json\\data_content_lable.json'

    # Mở và đọc nội dung tệp JSON
    with open(file_path, 'r', encoding='utf-8') as data_file:
        datas = json.load(data_file)

    # Xóa phần tử có "table": "answer"
    datas = [item for item in datas if item.get("table") != "answer"]

    # Ghi lại nội dung đã chỉnh sửa vào tệp JSON
    with open(file_path, 'w', encoding='utf-8') as data_file:
        json.dump(datas, data_file, indent=4, ensure_ascii=False)


# Gọi hàm
sql()

