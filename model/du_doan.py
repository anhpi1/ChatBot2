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
    for model,name_mode in zip(models, tables):
        du_doan_temp = du_doan_tong(cau_noi,model)
        #print(tkj.search_content_lable(table_name=name_mode, id=du_doan_temp))
        is_true_model = sp.load_model_true_false(name_mode+"_"+tkj.search_content_lable(table_name=name_mode, id=du_doan_temp)[0])
        temp.append(du_doan_temp)
        if(du_doan_tong(cau_noi,is_true_model)):
            #print (du_doan_temp)
            predict.append(du_doan_temp)
        else: 
             #print(0)
             predict.append(0)
        del is_true_model
    return predict,temp
#print(du_doan("what is the speed of comparator?",models))

def creater_report(models):
    question=[]
    with open('model\\data\\json\\data_test.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
        for data in datas:
            question.append(data["question"])  

    #print(question)
    

    matrix_du_doan=[]

    for row in question:
        du_doan_tempp,trash= du_doan(row,models)
        matrix_du_doan.append(du_doan_tempp)
        print(du_doan_tempp)
            
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
        tkj.tao_report(tabel, row,row_true)

def repair_train(incorrect_sentence_padded,false_answer,false_anwer_no_include_true_false,true_answer):
    for element_false,element_false_non_true_false,element_true, table in zip(false_answer,false_anwer_no_include_true_false,true_answer,tables):
        #print("/////////////////////////////////")
        if((element_false_non_true_false!=element_true) and (element_true>0)): 
            #print(table)                
            model = sp.load_model(table)
            sp.update_weights_on_incorrect_prediction( model, incorrect_sentence_padded, element_true)
        if(element_false!=element_true):
            if (element_true<=0):
                label=tkj.search_content_lable(table_name=table, id=element_false)[0]
                new_model_true_false = sp.load_model_true_false(table+"_"+label) 
                #print(table+"_"+label)  
                sp.update_weights_on_incorrect_prediction( new_model_true_false, incorrect_sentence_padded, 0)
            else:
                if(element_false<=0):
                    label=tkj.search_content_lable(table_name=table, id=element_true)[0]
                    new_model_true_false = sp.load_model_true_false(table+"_"+label)
                    
                    #print(table+"_"+label)  
                    sp.update_weights_on_incorrect_prediction(new_model_true_false, incorrect_sentence_padded, 1)
                    for t in tkj.creater_random_3_question(label):
                        
                        #print(table+"_"+label)  
                        sp.update_weights_on_incorrect_prediction( new_model_true_false, t, 0)
                else:
                    label=tkj.search_content_lable(table_name=table, id=element_false)[0]
                    #print(table+"_"+label)  
                    new_model_false = sp.load_model_true_false(table+"_"+label)
                    sp.update_weights_on_incorrect_prediction( new_model_false,incorrect_sentence_padded , 0)
                    label=tkj.search_content_lable(table_name=table, id=element_true)[0]
                    new_model_true = sp.load_model_true_false(table+"_"+label)
                    sp.update_weights_on_incorrect_prediction(new_model_true , incorrect_sentence_padded, 1)
                    for t in tkj.creater_random_3_question(label):
                        sp.update_weights_on_incorrect_prediction( new_model_true, t, 0)


def final_du_doan(question,models,true_answer=None):
    model_du_doan,temp=du_doan(question,models)
    if true_answer is not None:
        repair_train(question,model_du_doan,temp,true_answer)
    answer=sp.replace_positive(model_du_doan)
    print("ans:{}".format(answer))
    final_answer=[]
    con=sp.search_with_conditions_sqlserver(model_du_doan)
    if con != []:
        final_answer.append(con)
    for row in answer:
        con=sp.search_with_conditions_sqlserver(row)
        if con != []:
            final_answer.append(con)
            
    return final_answer




# models=[]
# for name_mode in tables:
#     new_model = sp.load_model(name_mode)
#     models.append(new_model)

# creater_report(models)
#print(len(final_du_doan("what is comparator",models)))
#creater_report(models)
