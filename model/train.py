import  suport as sp
from data.tham_so import tables
import tim_kiem_json as tkj
import random
import data.tham_so as ts
#print(tables)
data_train_x=[]
data_train_y=[]
for table in tables:
    tempx,tempy=tkj.search_data_question(table_name="{}".format(table), lable_name ="all" )
    # data_train_x.append(tempx)
    # data_train_y.append(tempy)
    questionn,_label=tkj.search_question_and_have_labe(table,"all")
    if(ts.co_train_du_lieu_test):
        print("true ////////////////////////////////////////////////?????????????????????????????????????/////////////////////////????????????????????????????????????????")
        # print(questionn)
        # print(_label)
        tempx=tempx+questionn
        tempy=tempy+_label
    number_of_outputs=len(tkj.search_name_lable(table_name="{}".format(table)))
    sp.train_TNN(table,  tempx, tempy, number_of_outputs+1)
    #print(table)
    #print(number_of_outputs)

for table in tables:  
    labels=tkj.search_name_lable(table_name=table )
    for label in labels:
        true_question , true_label=tkj.search_data_question(table, label)
        questionn,_label=tkj.search_question_and_have_labe(table,tkj.search_id_lable(table,label))
        if(ts.co_train_du_lieu_test):
 
            # print(table)
            # print(label)
            # print(tkj.search_id_lable(table,label))
            # print(true_question)
            # print(true_label)
            # print(questionn)
            # print(_label)
            #print(questionn)
            true_question=true_question+questionn
            true_label=true_label+_label
            print("true ////////////////////////////////////////////////?????????????????????????????????????/////////////////////////????????????????????????????????????????")
  
            # print(len(true_question))
            # print(len(true_label))
        for i in range(len(true_label)):
            if true_label[i] > 0:
                true_label[i] = 1
            
        random_label= [item for item in tkj.search_name_lable(table_name="all") if item != label ]
        random_choices_label = random.sample(random_label, ts.so_mau_false_x_20)
        # print(random_choices_table)
        # print(random_choices_label)
        false_question=[]
        false_lable= [0] * (ts.so_mau_false_x_20*20)
        for l in random_choices_label:
           # print(l)
            temp_question , temp=tkj.search_data_question(table_name="all",lable_name =l)
            # print(temp_question)
            # print(temp)
            false_question=false_question+temp_question
        # print("////////////////////////////////////")
        # print (label)
        # print(len(true_question+false_question))
        # print(len(true_label+false_lable))
        # print(" ")
        # print(len(true_question))
        # print(len(true_label))
        # print(" ")
        # print(len(false_question))
        # print(len(false_lable))
        #if(table=="techniques" and label=="compensate"):
        sp.train_TNN(table+"_"+label,true_question+false_question,true_label+false_lable , (1+1))
            
# print(len(data_train_x[1]))
# print(len(data_train_y[1]))
#print(data_train_x[0])
#print(data_train_y[0])

