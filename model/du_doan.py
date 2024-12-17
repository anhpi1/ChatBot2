from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import suport as sp
import tim_kiem_json as tkj
import numpy as np
import json
import os
import tensorflow as tf
from data.tham_so import file_word_list ,num_words_list,number_of_input,tables,weight_model

tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())



def du_doan_tong(input,model):
    with open(file_word_list, 'r') as json_file:
            word_index = json.load(json_file)
    tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
    tokenizer.word_index = word_index
    sequence = tokenizer.texts_to_sequences([input])
    padded_sequence = pad_sequences(sequence, maxlen=number_of_input)
    Ut = tf.constant(np.array(padded_sequence))
    predictions = model.predict(Ut, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

def du_doan(cau_noi,models):
    predict=[]
    temp=[]
    print (cau_noi)
    for model,name_mode in zip(models, tables):
        du_doan_temp = du_doan_tong(cau_noi,model)
        #print(tkj.search_content_lable(table_name=name_mode, id=du_doan_temp))
        #print(name_mode+"_"+tkj.search_content_lable(table_name=name_mode, id=du_doan_temp)[0])
        is_true_model = sp.load_model_true_false(name_mode+"_"+tkj.search_content_lable(table_name=name_mode, id=du_doan_temp)[0])
        temp.append(du_doan_temp)
        if(du_doan_tong(cau_noi,is_true_model)):
            #print (du_doan_temp)
            predict.append(du_doan_temp)
        else: 
             #print(0)
             predict.append(0)
        del is_true_model
    print(predict)
    print(temp)
    return predict,temp
#print(du_doan("what is the speed of comparator?",models))

def creater_report(models,name_report):
    question=[]
    with open('model\\data\\json\\data_test.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
        for data in datas:
            question.append(data["question"])  

    
    

    matrix_du_doan=[]

    for row in question:
        du_doan_tempp,trash= du_doan(row,models)
        matrix_du_doan.append(du_doan_tempp)
        #print(du_doan_tempp)
            
    matrix_du_doan_T=sp.transpose_matrix(matrix_du_doan) 
    data_file="model\\data\\json\\data_du_doan.json"
    matrix_du_doan_T_list=[[int(x) for x in y]for y in matrix_du_doan_T]
    print(matrix_du_doan_T)
    with open(data_file, 'w', encoding='utf-8') as output_file:
        json.dump(matrix_du_doan_T_list, output_file, indent=4, ensure_ascii=False)


    matrix_true_label=[]
    with open('model\\data\\json\\data_test.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
        for data in datas:
            temp=[]
            for tabel in tables:
                temp.append(data[tabel])  
            matrix_true_label.append(temp)
    matrix_true_label_T=sp.transpose_matrix(matrix_true_label)
    for row,row_true,tabel in zip(matrix_du_doan_T,matrix_true_label_T,tables):
        tkj.tao_report(tabel, row,row_true,name_report)



   
def repair_train(question_false,false_answer,false_answer_no_include_true_false,true_answer): 
    c=0                                  
    for dd,nb,t, table in zip(false_answer,false_answer_no_include_true_false,true_answer,tables):
        
        ddd=(nb==t)
        dnb=(dd==(t>0))
        dt=(t>0)
        label=sp.replace_space_with_underscore(tkj.search_content_lable(table_name=table, id=nb)[0])
        if(dnb and (not ddd) and dt):
            new_model_true_false = sp.load_model_true_false(table+"_"+label) 
            sp.update_weights_on_incorrect_prediction( new_model_true_false, question_false, 1)
            for t in tkj.creater_random_3_question(label):
                sp.update_weights_on_incorrect_prediction( new_model_true_false, question_false, 0)
            #print (4)
            new_model_true_false.save_weights(sp.replace_space_with_underscore(weight_model.format(table+"_"+label)))
            del new_model_true_false
        if((not dnb) and ddd and dt):
            model = sp.load_model(table)
            new_model_true_false = sp.load_model_true_false(table+"_"+label) 
            # print(question_false)
            print("update parent model{} and chill".format(c)) 
            
            #print(t)
            sp.update_weights_on_incorrect_prediction( new_model_true_false, question_false, 0)
            sp.update_weights_on_incorrect_prediction( model, question_false, t)
            model.save_weights(sp.replace_space_with_underscore(weight_model.format(table)))
            new_model_true_false.save_weights(sp.replace_space_with_underscore(weight_model.format(table+"_"+label)))
            del model,new_model_true_false
        if((not dnb)and (not ddd) and dt):
            print("update parent model{}".format(c))
            model = sp.load_model(table)
            sp.update_weights_on_incorrect_prediction( model, question_false, t)
            model.save_weights(sp.replace_space_with_underscore(weight_model.format(table)))
            del model
        c=c+1

import data.tham_so as ts
class ModelManager:
    def __init__(self):
        self.question_last=None
        self.model_du_doan = None
        self.model_du_doan_khong_gom_true_false = None

    def final_du_doan(self, question, models, true_answer=None):
        
        print("question: {}".format(question))
        if true_answer is not None:
            print("du doan    :{}".format(self.model_du_doan))
            print("true_answer:{}".format(true_answer))
            #print(self.model_du_doan_khong_gom_true_false)
            repair_train(self.question_last,self.model_du_doan,self.model_du_doan_khong_gom_true_false,true_answer)
        
        self.question_last=question
        self.model_du_doan, self.model_du_doan_khong_gom_true_false = du_doan(question, models)
        
        answer = sp.replace_positive(self.model_du_doan)
        
        final_answer = []
        print (answer)
        # Tìm kiếm trong SQL Server
        #con = sp.search_with_conditions_sqlserver(self.model_du_doan)
        con = [self.model_du_doan]
        if con:
            final_answer.append(con[0])
        for row in answer:
            con = sp.search_with_conditions_sqlserver(row)
            if con:
                final_answer.append(con[0])

        return final_answer

def check_model_loading(model_list):
    from tensorflow.keras.models import load_model
    failed_models = []

    for model_name in model_list:
        try:
            #print(f"Đang kiểm tra model: {model_name}")
            # Tạo đường dẫn đến trọng số
            
            # Giả sử trọng số lưu trong thư mục "weights"
            
            # Khởi tạo mô hình (cần thay thế bằng hàm khởi tạo đúng của bạn)
            model = sp.load_model_true_false(model_name)  # Thay thế bằng hàm của bạn
            
            # Cố gắng tải trọng số
            
           
        except Exception as e:
            # Nếu lỗi, ghi nhận model vào danh sách lỗi
            print(f"Lỗi khi tải model {model_name}")
            failed_models.append((model_name, str(e)))

    return failed_models
def ghi_cau_tra_loi(i,model_manager,true):

    with open('model\\data\\new model.txt','a', encoding='utf-8') as file:
        print(i)
        file.write("question:{}\n".format(i))
        answer = model_manager.final_du_doan(i, models)
        file.write("mess: {}".format(answer))
        file.write("\n")
        file.write("\n")



# Tải danh sách các mô hình
model_manager = ModelManager()
models = []
for name_mode in tables:  # Đảm bảo `tables` đã được định nghĩam
    new_model = sp.load_model(name_mode)
    models.append(new_model)

# # Thực hiện dự đoán
# answer = model_manager.final_du_doan("what is comparator ?", models)
# print(answer)

# print("Dự đoán có nhãn đúng:")
# answer = model_manager.final_du_doan("What are the static parameters of a comparator circuit?", models, [3, 3, 3, 3, 3, 3, 3, 3, 3])
# print(answer)


# #Xây dựng danh sách các mô hình từ tables và labels
# temp1 = []
# for table in tables:
#     #temp1.append(table)
#     labels = tkj.search_name_lable(table_name=table)
#     for label in labels:
#         temp1.append(table + "_" + label)

# # Kiểm tra tải các mô hình
# failed_models = check_model_loading(temp1)

# # Hiển thị các mô hình không tải được
# print("Các mô hình không tải được:")
# for model, error in failed_models:
#     print(f"- {model}: {error}")


creater_report(models,"new model")
temp=None
with open('model\\data\\json\\data_test.json', 'r', encoding='utf-8') as data_file:
    x =json.load(data_file)
    for row in x:     
        ghi_cau_tra_loi(row["question"],model_manager,temp)
        temp=[row["intent"],row["parameter"],row["structure"],row["operation"],row["components"],row["applications"],row["comparison"],row["techniques"],row["simulation"]]


# sum=0
# with open('model\\data\\json\\data_test.json', 'r', encoding='utf-8') as data_file:
#     x =json.load(data_file)
#     for table in tables:
#         labels=tkj.search_name_lable(table_name=table)
#         for label in labels:
#             c=0
#             for row in x:
#                 if(row[table]==tkj.search_id_lable(table_name=table, content=label)):
#                     c=c+1
#             sum=sum+c
#             print("table:{} label:{}".format(table,label))
#             print("so_nhan:{}".format(c))
# print(sum)
            

