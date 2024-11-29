import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import itertools
from transformers import pipeline
import pyodbc
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import re


# Đọc dữ liệu từ file
input = []
with open('data_train\input_train\content_question-test.ta', 'r') as f:
    for line in f:
        input.append(line.strip())  # Loại bỏ ký tự newline ở cuối dòng
# with open('input.txt', 'r') as f:
#     for line in f:
#         input.append(line.strip())  # Loại bỏ ký tự newline ở cuối dòng

# Khởi tạo danh sách nhãn
nhan_o_dang_chu = []
topics = ['question_type', 'question_intent', 'concept1', 'concept2', 'structure', 'performance_metric', 'design_techniques', 'applications', 'components', 'testing_simulations_tools']

# Kết nối với cơ sở dữ liệu
conn = pyodbc.connect(
    r'DRIVER={SQL Server};'
    r'SERVER=DESKTOP-1MU0IU3\SQLEXPRESS;'
    r'DATABASE=comparator;'
    r'UID=;'
    r'PWD='
)
cursor = conn.cursor()

# Lấy dữ liệu từ cơ sở dữ liệu
for i in range(0, 3):  # Thay đổi dải số nếu cần lấy nhiều topic
    cursor.execute("SELECT content FROM {}".format(topics[i]))
    results = cursor.fetchall()
    column_data = [row[0] for row in results]
    nhan_o_dang_chu.append(column_data)

cursor.close()

# Hàm phân loại văn bản
def classify_text(text, classifier, candidate_labels):
    c=0
    du_doan = []
    for line in text:
        c=c+1
        print(c)
        result = classifier(line, candidate_labels=candidate_labels, multi_label=False, truncation=True, max_length=32)
        du_doan.append(result['labels'][0])
    return du_doan

for i in range(2,3):
    du_doan_dang_so = []
    start_time_all = time.time()
    # Phân loại văn bản với các mô hình khác nhau
    print("model{}".format(i))
    start_time1 = time.time()
    du_doan_dang_chu = []
    classifier1 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    du_doan_dang_chu.append(classify_text(input, classifier1, nhan_o_dang_chu[i]))
    del classifier1
    end_time1 = time.time()
    elapsed_time_1 = end_time1 - start_time1
    print("mo hinh 1 hoanh thanh")


    start_time2 = time.time()
    classifier2 = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    du_doan_dang_chu.append(classify_text(input, classifier2, nhan_o_dang_chu[i]))
    del classifier2
    end_time2 = time.time()
    elapsed_time_2 = end_time2 - start_time2
    print("mo hinh 2 hoanh thanh")


    start_time3 = time.time()
    classifier3 = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base")
    du_doan_dang_chu.append(classify_text(input, classifier3, nhan_o_dang_chu[i]))
    del classifier3
    end_time3 = time.time()
    elapsed_time_3 = end_time3 - start_time3
    print("mo hinh 3 hoanh thanh")


    start_time4 = time.time()
    classifier4 = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    du_doan_dang_chu.append(classify_text(input, classifier4, nhan_o_dang_chu[i]))
    del classifier4
    end_time4 = time.time()
    elapsed_time_4 = end_time4 - start_time4
    print("mo hinh 4 hoanh thanh")


    start_time5= time.time()
    classifier5 = pipeline("zero-shot-classification", model="Sahajtomar/German_Zeroshot")
    du_doan_dang_chu.append(classify_text(input, classifier5, nhan_o_dang_chu[i]))
    del classifier5
    end_time5 = time.time()
    elapsed_time_5 = end_time5 - start_time5
    print("mo hinh 5 hoanh thanh")
    # Chuyển đổi dự đoán thành id từ cơ sở dữ liệu
    
    cursor = conn.cursor()
    print("du_doan_dang_chu:{}".format(du_doan_dang_chu))
    for line in du_doan_dang_chu:
        temp = []
        for element in line:
            
            if (element== None):
                temp.append(0)
            else:
                cursor.execute("SELECT id FROM {} WHERE CONVERT(VARCHAR(MAX), content) = '{}';".format(topics[i], element))
                result = cursor.fetchall()
                temp.append(result[0][0])
        du_doan_dang_so.append(temp)
    print("du_doan_dang_so:{}".format(du_doan_dang_so))
    with open("data_train\\report\\du_doan_dang_so_{}.log".format(i), "w", encoding="utf-8") as log_file:
        for line in du_doan_dang_so:
            log_file.write(str(line) + "\n")  # Chuyển mỗi phần tử thành chuỗi và thêm dòng mới

    cursor.close()
    name=['facebook/bart-large-mnli','MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli','knowledgator/comprehend_it-base','MoritzLaurer/deberta-v3-large-zeroshot-v2.0','Sahajtomar/German_Zeroshot']
    for du_doan,nmodel in zip(du_doan_dang_so,name):
        with open("data_train\output_train\o{}_test.ta".format(i), "r", encoding="utf-8") as output_file:
            lines = output_file.readlines()
            test = [int(x) for x in [line.strip() for line in lines]]
        # with open("output_{}.txt".format(i), "r", encoding="utf-8") as output_file:
        #     lines = output_file.readlines()
        #     test = [int(x) for x in [line.strip() for line in lines]]
        print("test:{}".format(test))
        end_time_all = time.time()
        elapsed_time_all = end_time_all - start_time_all
        print("time run:{}".format(elapsed_time_all))
        #Tính toán các chỉ số đánh giá
        safe_model_name = re.sub(r'[\/:*?"<>|]', '_', nmodel)

        with open("data_train\\report\\log_{}_{}.log".format(i,safe_model_name), "w", encoding="utf-8") as log_file:
            # Confusion Matrix
            cm = confusion_matrix(du_doan, test, labels=[label for label in range(len(topics[i]))])
            
            # Các chỉ số đánh giá khác
            accuracy = accuracy_score(du_doan, test)
            precision = precision_score(du_doan, test, average='macro')
            recall = recall_score(du_doan, test, average='macro')
            f1 = f1_score(du_doan, test, average='macro')
            
            # Ghi vào file log
            log_file.write(f"Class: {i}\n\n")
            log_file.write("Confusion Matrix:\n")
            log_file.write(f"{cm}\n\n")
            log_file.write(f"Accuracy: {accuracy:.4f}\n")
            log_file.write(f"Precision: {precision:.4f}\n")
            log_file.write(f"Recall: {recall:.4f}\n")
            log_file.write(f"F1 Score: {f1:.4f}\n")
            log_file.write(f"time model 1:  {elapsed_time_1:.4f}\n")
            log_file.write(f"time model 2:  {elapsed_time_2:.4f}\n")
            log_file.write(f"time model 3:  {elapsed_time_3:.4f}\n")
            log_file.write(f"time model 4:  {elapsed_time_4:.4f}\n")
            log_file.write(f"time model 5:  {elapsed_time_5:.4f}\n")
            log_file.write(f"time model all:  {elapsed_time_all:.4f}\n")
            log_file.write(f"du_doan_dang_so:  {du_doan_dang_so}\n")


conn.close()


# [[1, 1, 1, 1, 6, 6, 6, 4, 4, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 6, 6, 1, 4, 7, 7, 1, 7, 7, 1, 1, 1, 6, 1, 4, 7, 1, 1, 4, 3, 1, 1, 1, 6, 1, 1, 1, 1, 1, 3, 6, 6, 6, 8, 1, 6, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 6, 6, 1, 6, 1, 1, 8, 4, 1, 1, 1, 1, 1, 4, 1, 4, 1, 6, 1, 1, 4, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 3, 3, 3, 1, 6, 6, 4, 1, 1, 3, 7, 1, 1, 6, 1, 1, 6, 6, 6], 
#  [8, 1, 9, 9, 1, 8, 7, 7, 7, 9, 5, 9, 9, 9, 9, 9, 7, 9, 9, 8, 6, 8, 4, 8, 9, 1, 8, 9, 5, 9, 9, 5, 9, 9, 9, 9, 1, 6, 7, 1, 1, 8, 1, 5, 9, 9, 8, 9, 3, 9, 8, 5, 2, 5, 9, 1, 9, 1, 1, 1, 7, 9, 9, 9, 9, 9, 9, 1, 6, 9, 6, 7, 1, 9, 9, 8, 1, 5, 9, 9, 5, 1, 5, 1, 1, 9, 9, 1, 9, 9, 9, 9, 9, 3, 8, 7, 9, 9, 7, 9, 3, 9, 9, 5, 4, 9, 9, 3, 8, 8, 9, 9, 3, 3, 9, 5, 9], 
#  [2, 6, 6, 8, 6, 5, 7, 7, 7, 8, 6, 8, 7, 5, 6, 7, 6, 7, 9, 6, 8, 4, 8, 8, 2, 8, 5, 8, 2, 5, 2, 7, 5, 8, 8, 8, 8, 8, 4, 8, 8, 8, 9, 8, 3, 5, 5, 5, 5, 5, 6, 8, 8, 8, 8, 8, 5, 7, 7, 8, 8, 7, 8, 8, 6, 4, 7, 7, 8, 8, 8, 8, 8, 8, 5, 8, 5, 8, 4, 6, 8, 8, 6, 8, 5, 7, 3, 8, 8, 8, 8, 7, 3, 3, 8, 4, 5, 4, 8, 3, 8, 8, 8, 5, 6, 5, 4], 
#  [2, 1, 1, 1, 1, 1, 7, 7, 7, 1, 6, 1, 1, 5, 1, 1, 7, 1, 9, 1, 1, 8, 4, 1, 2, 1, 1, 5, 1, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 5, 1, 5, 5, 9, 1, 1, 1, 1, 1, 8, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 2, 5, 1, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 8, 1, 1, 1, 7, 3, 3, 1, 1, 1, 4, 1, 1, 3, 8, 8, 1, 5, 1, 1, 1, 1], 
#  [2, 2, 2, 1, 1, 5, 2, 2, 5, 1, 6, 1, 1, 9, 1, 1, 9, 1, 9, 1, 2, 2, 4, 1, 1, 2, 1, 9, 5, 2, 2, 5, 1, 1, 5, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3, 5, 5, 5, 2, 5, 2, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 2, 1, 1, 6, 1, 5, 9, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 5, 1, 1, 1, 9, 9, 2, 1, 2, 2, 1, 3, 1, 1, 1, 1, 2, 9, 1, 1, 1, 5, 4, 1, 1, 3, 2, 2, 1, 5, 1, 2, 5, 5, 5]]

# [[[(1,)], [(1,)], [(1,)], [(1,)], [(6,)], [(6,)], [(6,)], [(4,)], [(4,)], [(1,)], [(6,)], [(1,)], [(6,)], [(6,)], [(6,)], [(6,)], [(6,)], [(6,)], [(1,)], [(6,)], [(6,)], [(1,)], [(4,)], [(7,)], [(7,)], [(1,)], [(7,)], [(7,)], [(1,)], [(1,)], [(1,)], [(6,)], [(1,)], [(4,)], [(7,)], [(1,)], [(1,)], [(4,)], [(3,)], [(1,)], [(1,)], [(1,)], [(6,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(3,)], [(6,)], [(6,)], [(6,)], [(8,)], [(1,)], [(6,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(7,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(6,)], [(6,)], [(1,)], [(6,)], [(1,)], [(1,)], [(8,)], [(4,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(4,)], [(1,)], [(4,)], [(1,)], [(6,)], [(1,)], [(1,)], [(4,)], [(3,)], [(1,)], [(1,)], [(1,)], [(1,)], [(4,)], [(1,)], [(1,)], [(1,)], [(1,)], [(3,)], [(3,)], [(3,)], [(1,)], [(6,)], [(6,)], [(4,)], [(1,)], [(1,)], [(3,)], [(7,)], [(1,)], [(1,)], [(6,)], [(1,)], [(1,)], [(6,)], [(6,)], [(6,)]], 
#  [[(8,)], [(1,)], [(9,)], [(9,)], [(1,)], [(8,)], [(7,)], [(7,)], [(7,)], [(9,)], [(5,)], [(9,)], [(9,)], [(9,)], [(9,)], [(9,)], [(7,)], [(9,)], [(9,)], [(8,)], [(6,)], [(8,)], [(4,)], [(8,)], [(9,)], [(1,)], [(8,)], [(9,)], [(5,)], [(9,)], [(9,)], [(5,)], [(9,)], [(9,)], [(9,)], [(9,)], [(1,)], [(6,)], [(7,)], [(1,)], [(1,)], [(8,)], [(1,)], [(5,)], [(9,)], [(9,)], [(8,)], [(9,)], [(3,)], [(9,)], [(8,)], [(5,)], [(2,)], [(5,)], [(9,)], [(1,)], [(9,)], [(1,)], [(1,)], [(1,)], [(7,)], [(9,)], [(9,)], [(9,)], [(9,)], [(9,)], [(9,)], [(1,)], [(6,)], [(9,)], [(6,)], [(7,)], [(1,)], [(9,)], [(9,)], [(8,)], [(1,)], [(5,)], [(9,)], [(9,)], [(5,)], [(1,)], [(5,)], [(1,)], [(1,)], [(9,)], [(9,)], [(1,)], [(9,)], [(9,)], [(9,)], [(9,)], [(9,)], [(3,)], [(8,)], [(7,)], [(9,)], [(9,)], [(7,)], [(9,)], [(3,)], [(9,)], [(9,)], [(5,)], [(4,)], [(9,)], [(9,)], [(3,)], [(8,)], [(8,)], [(9,)], [(9,)], [(3,)], [(3,)], [(9,)], [(5,)], [(9,)]], 
#  [[(2,)], [(6,)], [(6,)], [(8,)], [(6,)], [(5,)], [(7,)], [(7,)], [(7,)], [(8,)], [(6,)], [(8,)], [(7,)], [(5,)], [], [(6,)], [(7,)], [(6,)], [(7,)], [(9,)], [(6,)], [(8,)], [(4,)], [(8,)], [(8,)], [(2,)], [(8,)], [], [(5,)], [(8,)], [(2,)], [(5,)], [(2,)], [(7,)], [(5,)], [(8,)], [(8,)], [], [], [(8,)], [(8,)], [(8,)], [(4,)], [(8,)], [(8,)], [(8,)], [(9,)], [(8,)], [(3,)], [(5,)], [(5,)], [(5,)], [(5,)], [(5,)], [(6,)], [(8,)], [(8,)], [(8,)], [(8,)], [(8,)], [(5,)], [(7,)], [(7,)], [(8,)], [(8,)], [(7,)], [(8,)], [(8,)], [(6,)], [(4,)], [(7,)], [(7,)], [], [], [(8,)], [(8,)], [(8,)], [(8,)], [(8,)], [(8,)], [(5,)], [(8,)], [(5,)], [(8,)], [(4,)], [(6,)], [(8,)], [(8,)], [(6,)], [(8,)], [(5,)], [(7,)], [], [(3,)], [(8,)], [(8,)], [(8,)], [(8,)], [(7,)], [(3,)], [(3,)], [(8,)], [(4,)], [(5,)], [(4,)], [], [(8,)], [(3,)], [(8,)], [(8,)], [(8,)], [(5,)], [], [], [(6,)], [(5,)], [(4,)]], 
#  [[(2,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(7,)], [(7,)], [(7,)], [(1,)], [(6,)], [(1,)], [(1,)], [(5,)], [(1,)], [(1,)], [(7,)], [(1,)], [(9,)], [(1,)], [(1,)], [(8,)], [(4,)], [(1,)], [], [(2,)], [(1,)], [(1,)], [(5,)], [(1,)], [(1,)], [(5,)], [], [(1,)], [(5,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [], [], [(1,)], [(1,)], [(1,)], [(7,)], [(5,)], [(1,)], [(5,)], [], [(5,)], [(9,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(8,)], [], [(1,)], [], [], [], [], [(1,)], [(6,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(2,)], [(5,)], [(1,)], [(1,)], [(5,)], [(1,)], [(5,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(3,)], [(8,)], [(1,)], [(1,)], [(1,)], [(7,)], [(3,)], [(3,)], [(1,)], [(1,)], [(1,)], [(4,)], [(1,)], [(1,)], [(3,)], [(8,)], [(8,)], [(1,)], [(5,)], [(1,)], [(1,)], [(1,)], [], [(1,)]], 
#  [[(2,)], [(2,)], [(2,)], [(1,)], [(1,)], [(5,)], [(2,)], [(2,)], [(5,)], [(1,)], [(6,)], [(1,)], [(1,)], [(9,)], [(1,)], [(1,)], [(9,)], [(1,)], [(9,)], [(1,)], [(2,)], [(2,)], [(4,)], [(1,)], [(1,)], [(2,)], [(1,)], [(9,)], [(5,)], [(2,)], [(2,)], [(5,)], [(1,)], [(1,)], [(5,)], [(9,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(5,)], [(3,)], [(5,)], [(5,)], [(5,)], [(2,)], [(5,)], [(2,)], [(1,)], [(1,)], [(1,)], [(1,)], [(1,)], [(5,)], [(1,)], [(1,)], [(1,)], [(1,)], [(2,)], [(1,)], [(1,)], [(6,)], [(1,)], [(5,)], [(9,)], [(1,)], [(1,)], [(5,)], [(1,)], [(1,)], [(5,)], [(1,)], [(1,)], [(5,)], [(1,)], [(5,)], [(1,)], [(1,)], [(1,)], [(9,)], [(9,)], [(2,)], [(1,)], [(2,)], [(2,)], [(1,)], [(3,)], [(1,)], [(1,)], [(1,)], [(1,)], [(2,)], [(9,)], [(1,)], [(1,)], [(1,)], [(5,)], [(4,)], [(1,)], [(1,)], [(3,)], [(2,)], [(2,)], [(1,)], [(5,)], [(1,)], [(2,)], [(5,)], [(5,)], [(5,)]]]


# [[4, 11, 11, 12, 11, 8, 4, 8, 8, 4, 8, 4, 5, 5, 10, 8, 5, 10, 4, 10, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 10, 8, 8, 8, 4, 4, 4, 8, 4, 8, 8, 8, 8, 8, 8, 8, 15, 7, 2, 15, 7, 2, 10, 8, 8, 8, 11, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 11, 10, 8, 10, 10, 8, 8, 8, 17, 17, 8, 8, 8, 8, 8, 1, 2, 10, 10, 8, 8, 18, 8, 8, 8, 8, 11, 8, 8, 10, 11, 2, 8, 8], 
#  [4, 10, 10, 15, 11, 8, 4, 8, 8, 4, 8, 4, 8, 8, 8, 7, 8, 8, 4, 8, 11, 4, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 11, 8, 14, 8, 8, 8, 4, 4, 4, 8, 8, 8, 7, 8, 11, 3, 4, 4, 4, 8, 4, 8, 8, 8, 4, 8, 8, 11, 8, 4, 8, 8, 8, 8, 7, 8, 15, 8, 11, 8, 8, 8, 8, 11, 8, 8, 11, 8, 11, 8, 9, 15, 8, 11, 7, 8, 8, 17, 17, 17, 11, 8, 11, 8, 8, 8, 7, 7, 7, 8, 8, 7, 8, 8, 2, 8, 11, 11, 4, 10, 8, 11, 8, 8], 
#  [4, 4, 4, 4, 4, 10, 4, 13, 8, 4, 14, 4, 11, 0, 10, 10, 2, 13, 10, 10, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 13, 8, 7, 13, 4, 4, 4, 8, 8, 8, 10, 8, 13, 3, 4, 4, 4, 8, 4, 8, 8, 2, 7, 2, 15, 13, 15, 7, 2, 15, 7, 2, 10, 8, 10, 4, 13, 8, 8, 13, 8, 8, 8, 3, 8, 8, 8, 8, 9, 10, 8, 11, 10, 8, 17, 18, 17, 17, 13, 17, 13, 1, 2, 1, 0, 0, 10, 13, 8, 10, 13, 8, 5, 2, 2, 13, 13, 8, 8, 13, 8, 13], 
#  [4, 4, 4, 4, 12, 15, 4, 8, 8, 14, 10, 4, 0, 9, 10, 8, 8, 8, 14, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 8, 8, 8, 0, 8, 8, 8, 4, 4, 4, 8, 4, 8, 8, 8, 7, 2, 15, 0, 15, 7, 2, 15, 7, 2, 0, 8, 15, 8, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 11, 8, 14, 0, 8, 17, 8, 17, 17, 8, 8, 8, 8, 8, 1, 0, 17, 0, 8, 8, 0, 8, 8, 5, 8, 0, 8, 12, 8, 8, 4, 8, 8], 
#  [4, 14, 14, 4, 4, 8, 4, 8, 8, 4, 2, 4, 8, 8, 10, 8, 13, 8, 14, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 4, 14, 4, 8, 8, 8, 2, 8, 8, 3, 4, 4, 8, 8, 4, 8, 8, 8, 4, 8, 15, 2, 15, 8, 2, 15, 8, 8, 8, 8, 2, 4, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 17, 18, 17, 17, 8, 8, 8, 8, 8, 1, 2, 8, 2, 8, 8, 8, 8, 8, 8, 8, 4, 8, 4, 10, 11, 2, 8, 8]]

# [[1, 1, 1, 1, 6, 6, 6, 4, 4, 1, 6, 1, 6, 6, 6, 6, 6, 6, 1, 6, 6, 1, 4, 7, 7, 1, 7, 7, 1, 1, 1, 6, 1, 4, 7, 1, 1, 4, 3, 1, 1, 1, 6, 1, 1, 1, 1, 1, 3, 6, 6, 6, 8, 1, 6, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 6, 6, 1, 6, 1, 1, 8, 4, 1, 1, 1, 1, 1, 4, 1, 4, 1, 6, 1, 1, 4, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 3, 3, 3, 1, 6, 6, 4, 1, 1, 3, 7, 1, 1, 6, 1, 1, 6, 6, 6], 
#  [8, 1, 9, 9, 1, 8, 7, 7, 7, 9, 5, 9, 9, 9, 9, 9, 7, 9, 9, 8, 6, 8, 4, 8, 9, 1, 8, 9, 5, 9, 9, 5, 9, 9, 9, 9, 1, 6, 7, 1, 1, 8, 1, 5, 9, 9, 8, 9, 3, 9, 8, 5, 2, 5, 9, 1, 9, 1, 1, 1, 7, 9, 9, 9, 9, 9, 9, 1, 6, 9, 6, 7, 1, 9, 9, 8, 1, 5, 9, 9, 5, 1, 5, 1, 1, 9, 9, 1, 9, 9, 9, 9, 9, 3, 8, 7, 9, 9, 7, 9, 3, 9, 9, 5, 4, 9, 9, 3, 8, 8, 9, 9, 3, 3, 9, 5, 9], 
#  [2, 6, 6, 8, 6, 5, 7, 7, 7, 8, 6, 8, 7, 5, 0, 6, 7, 6, 7, 9, 6, 8, 4, 8, 8, 2, 8, 0, 5, 8, 2, 5, 2, 7, 5, 8, 8, 0, 0, 8, 8, 8, 4, 8, 8, 8, 9, 8, 3, 5, 5, 5, 5, 5, 6, 8, 8, 8, 8, 8, 5, 7, 7, 8, 8, 7, 8, 8, 6, 4, 7, 7, 0, 0, 8, 8, 8, 8, 8, 8, 5, 8, 5, 8, 4, 6, 8, 8, 6, 8, 5, 7, 0, 3, 8, 8, 8, 8, 7, 3, 3, 8, 4, 5, 4, 0, 8, 3, 8, 8, 8, 5, 0, 0, 6, 5, 4], 
#  [2, 1, 1, 1, 1, 1, 7, 7, 7, 1, 6, 1, 1, 5, 1, 1, 7, 1, 9, 1, 1, 8, 4, 1, 0, 2, 1, 1, 5, 1, 1, 5, 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 7, 5, 1, 5, 0, 5, 9, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0, 0, 1, 6, 1, 1, 1, 1, 1, 1, 1, 2, 5, 1, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 8, 1, 1, 1, 7, 3, 3, 1, 1, 1, 4, 1, 1, 3, 8, 8, 1, 5, 1, 1, 1, 0, 1], 
#  [2, 2, 2, 1, 1, 5, 2, 2, 5, 1, 6, 1, 1, 9, 1, 1, 9, 1, 9, 1, 2, 2, 4, 1, 1, 2, 1, 9, 5, 2, 2, 5, 1, 1, 5, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3, 5, 5, 5, 2, 5, 2, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 2, 1, 1, 6, 1, 5, 9, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 5, 1, 1, 1, 9, 9, 2, 1, 2, 2, 1, 3, 1, 1, 1, 1, 2, 9, 1, 1, 1, 5, 4, 1, 1, 3, 2, 2, 1, 5, 1, 2, 5, 5, 5]]