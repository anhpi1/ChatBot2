

import json


def search_content_lable(table_name=None, id=None):
    #print("contenlabe: {}_{}".format(table_name,id))
    file_path="model\\data\\json\\data_content_lable.json"
    try:
        # Đọc file JSON
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Danh sách kết quả trả về
        results = []
        

        # Lọc dữ liệu theo điều kiện
        for table in data:
            if table_name and table["table"] != table_name:
                continue

            for row in table["data"]:
                if id is not None and row["id"] != id:
                    continue
                results.append(row["content"])
                

        return results

    except FileNotFoundError:
        return {"error": "File JSON không tồn tại!"}
    except json.JSONDecodeError:
        return {"error": "File JSON không hợp lệ!"}

def search_id_lable(table_name=None, content=None):
    file_path="model\\data\\json\\data_content_lable.json"
    # Đọc file JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
        results = []
        for data in datas:
            if(content=="all"):
                for row in data["data"]:
                    results.append(row["id"])
            if(data["table"]==table_name):
                for row in data["data"]:
                    if(row["content"]==content):
                        return row["id"]

    
    
    return results
#print(search_id_lable("intent", "definition"))

def search_data_question(table_name=None, lable_name =None ):
    file_path="model\\data\\json\\data_train.json"
    if lable_name is None or table_name is None:
        return
    try:
        # Đọc file JSON
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Danh sách kết quả trả về
        results = []
        results_true_label = []
        # Lọc dữ liệu theo điều kiện
        for table in data:
            if (table_name!="all"):
                if table["table"] != table_name:
                    continue
            

            for label in table["data"]:
                if (lable_name =="all" ):
                    for row in label["content"]:
                        results.append(row)
                        results_true_label.append(label["true_label"])
                    continue
                if label["label"] != lable_name:
                    continue
                for row in label["content"]:
                    results.append(row)
                    results_true_label.append(label["true_label"])

        return results,results_true_label

    except FileNotFoundError:
        return {"error": "File JSON không tồn tại!"}
    except json.JSONDecodeError:
        return {"error": "File JSON không hợp lệ!"}
# results,results_true_label=search_data_question("intent","definition")
# print(results)
# print(results_true_label)
def search_name_lable(table_name=None):
    file_path="model\\data\\json\\data_content_lable.json"
    if table_name is None:
        return
        
    try:
        # Đọc file JSON
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Danh sách kết quả trả về
        results = []

        # Lọc dữ liệu theo điều kiện
        for table in data:

            if table_name!="all":
                
                if table["table"]!=table_name:
                    continue
            for row in table["data"]:
                if row["content"] is None:
                    continue
                results.append(row["content"]) 
                continue

        return results

    except FileNotFoundError:
        return {"error": "File JSON không tồn tại!"}
    except json.JSONDecodeError:
        return {"error": "File JSON không hợp lệ!"}

from data.tham_so import report_train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def tao_report(name, output_test,predictions,name_report):
    with open(report_train.format(name,name_report), "w", encoding="utf-8") as file:
        accuracy = accuracy_score(output_test, predictions)
        precision = precision_score(output_test, predictions, average='macro',zero_division=0)
        recall = recall_score(output_test, predictions, average='macro',zero_division=0)
        f1 = f1_score(output_test, predictions, average='macro',zero_division=0)
        conf_matrix = confusion_matrix(output_test, predictions)   
        file.write("\n")
        file.write(f"Model: {name}")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write("Confusion Matrix:\n")
        
        # Chuyển confusion matrix thành chuỗi và ghi vào tệp
        for row in conf_matrix:
            file.write(" ".join(map(str, row)) + "\n")

        file.write("\n//////////////////////////////////////////////////////////////////////////////////////////////\n\n")

import random
def creater_random_3_question(no_include_labe):
    labels= [item for item in search_name_lable(table_name="all") if item != no_include_labe]
    random_label = random.sample(labels, 3)
    listt=[]
    for r in random_label:
        temp,trash=search_data_question(table_name="all",lable_name =r)
        listt=listt+temp
    return random.sample(listt, 3)

def search_question_and_have_labe(table,label ):
    question=[]
    _lable=[]
    
    if (table==None or label==None):
        return
    if(label =="all"):
        file_path="model\data\json\data_test.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for row in data:
                if(row[table]>0):
                    question.append(row["question"])
                    _lable.append(row[table])
        
        
        return question,_lable
    file_path="model\data\json\data_test.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for row in data:
            if(row[table]==label):
                question.append(row["question"])
                _lable.append(row[table])
    # print("1.{}".format(label))
    # print("2.{}".format(question))
    # print("3.{}".format(_lable))
    return question,_lable
# questionn,_lable=search_question_and_have_labe("intent",11)
# print(questionn)
# print(_lable)

#print(creater_random_3_question("definition"))
#print(search_content_lable(table_name="applications", id=2))
# question , true_label=search_data_question(table_name="all", lable_name ="all")
# print(len(question))
# print(len(true_label))
#print((search_id_lable(table_name="intent",content="definition")))
# import data.tham_so as ts 
# for row in ts.tables:
#     for i in range(5):
#         print(len(search_name_lable(table_name=row)))

