import json
import data.tham_so as ts
temp=[]
with open('model/data/json/data_content_lable.json', 'r', encoding='utf-8') as data_file:
    x =json.load(data_file)
    for table in ts.tables:
        for row in x:
            if row["table"]==table:
                for label in row["data"]:
                    if label["content"]==None:continue
                    temp.append(label["content"])
                    print(label["content"])
print(len(temp))  












     