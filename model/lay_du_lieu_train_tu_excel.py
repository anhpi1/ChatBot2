import json
from openpyxl import load_workbook
from data.tham_so import tables
import tim_kiem_json as tkj

def read_row(file_path, sheet_name, column, start_row, end_row):    
    # Mở file Excel
    workbook = load_workbook(file_path)
    
    # Chọn sheet
    sheet = workbook[sheet_name]
    
    # Đọc dữ liệu từ cột trong khoảng được chỉ định
    data_column = []
    for row in range(start_row, end_row + 1):
        cell_value = sheet[f"{column}{row}"].value  # Lấy giá trị ô
        data_column.append(cell_value)
    return data_column 
 

def read_file_train(file_path, sheet_name, column, start_row, end_row, output_file):    
    data_column=read_row(file_path, sheet_name, column, start_row, end_row)
    with open('model\\data\\json\\data_column.json', 'w', encoding='utf-8') as data_file:
        json.dump(data_column, data_file, indent=4, ensure_ascii=False)
    # Ghi dữ liệu thành JSON mỗi 20 dòng
    chunks = [data_column[i:i + 20] for i in range(0, len(data_column), 20)]
    
    #print(len(chunks))
    # chunk_lengths = [len(chunk) for chunk in chunks]
    # print(chunk_lengths)
    c=0
    datas=[]
    for table in tables:
        with open('model\\data\\json\\data_table_{}.json'.format(table), 'r', encoding='utf-8') as data_file:
            labels =json.load(data_file)
        data=[]
        for label in labels:
            if(label==None):
                continue
            true_label=tkj.search_id_lable(table_name="{}".format(table),content="{}".format(label))
            data.append({
                "label":label,
                "true_label": true_label,
                "content":chunks[c]
                })
            c=c+1
            #print(table)
            #print(label)
            #print(c)
        datas.append({
            "table":table,
            "data":data
            })
    with open(output_file, 'w', encoding='utf-8') as data_file:
        json.dump(datas, data_file, indent=4, ensure_ascii=False)
    return data_column

#read_row(file_path, sheet_name, column, start_row, end_row)
import numpy as np
def read_file_test(file_path,sheet_name,start_row,end_row):
    alphabet=['B','C','D','E','F','G','H','I','J','K']
    table=["question"]+tables
    # Tạo ma trận từ các cột trong bảng tính
    # Tạo ma trận ban đầu (có thể chứa số, None và câu nói)
    matrix = []
    for column in alphabet:
        matrix.append(read_row(file_path, sheet_name, column, start_row, end_row))

    # Thay thế None bằng 0 trong ma trận
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] =='NULL':
                matrix[i][j] = 0
            elif isinstance(matrix[i][j], str):  # Kiểm tra nếu là chuỗi, giữ nguyên
                continue
            else:  # Nếu là số, chuyển sang kiểu int
                matrix[i][j] = int(matrix[i][j])

    # Chuyển vị ma trận (hàng thành cột, cột thành hàng)
    matrix_T = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    result=[]
    for row in matrix_T:
        row_dict = {name: d for name, d in zip(table, row)}  # Tạo dictionary từ cặp (name, d)
        result.append(row_dict)
    data_file="model\\data\\json\\data_test.json"
    with open(data_file, 'w', encoding='utf-8') as data_file:
        json.dump(result, data_file, indent=4, ensure_ascii=False)

    #print(matrix_T)
    
    
   
# Gọi hàm
file_path = "model\\data\\xlxs\\question.xlsx"  # Đường dẫn tới file Excel
sheet_name = "train_question"  # Tên sheet
column = "B"  # Cột cần đọc
start_row = 3  # Hàng bắt đầu
end_row = 1922  # Hàng kết thúc
output_file = "model\\data\\json\\data_train.json"  # Tên file JSON đầu ra

read_file_train(file_path, sheet_name, column, start_row, end_row, output_file)
# print(len(read_row(file_path, sheet_name, column, start_row, end_row)))
sheet_name = "question"
start_row = 3  # Hàng bắt đầu
end_row = 147
read_file_test(file_path,sheet_name,start_row,end_row)
 
 
