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

    # Đóng kết nối
    cursor.close()
    connection.close()

# Gọi hàm
sql()

