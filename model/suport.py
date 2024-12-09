def transpose_matrix(matrix):
    # Khởi tạo ma trận mới với số cột và số hàng hoán đổi
    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transposed

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import json
import os
import shutil
from tensorflow.keras.optimizers import Adam
from data.tham_so import file_word_list ,num_words_list,weight_model,number_of_input
from tensorflow.keras.utils import to_categorical


# Sử dụng tất cả lõi CPU có sẵn
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())


# Hàm tạo transformer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Self-attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    # Feed-forward layer
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    return ff_output

def create_model(number_of_outputs):
    # Xây dựng mô hình transformer
    input_layer = Input(shape=(number_of_input,))
    embedding_layer = Embedding(input_dim=num_words_list, output_dim=128, input_length=number_of_input)(input_layer)
    x = transformer_encoder(embedding_layer, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(number_of_outputs, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def train_TNN(k, file_input_train, file_output_train, number_of_outputs):
    name_mode=replace_space_with_underscore(k)
    input_padded = convert_to_pad(file_input_train)

    # Nhãn là mảng số nguyên
    labels = np.array(file_output_train)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(input_padded, labels, test_size=0.2, random_state=42)
    
    # Xây dựng mô hình
    model = create_model(number_of_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Định nghĩa callback early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=ts.so_lan_loss_k_thay_doi, restore_best_weights=True)

    # Huấn luyện mô hình
    model.fit(X_train, y_train, epochs=6000, batch_size=ts.so_mau_train, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

    # Dự đoán từ mô hình
    predictions = model.predict(X_test, verbose=0).argmax(axis=1)

    # y_test không cần chuyển đổi, nó là nhãn số nguyên rồi
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)

    # In kết quả
    print(f"Model: {name_mode}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

    # Lưu trọng số mô hình
    model.save_weights(weight_model.format(name_mode))
    del model

def convert_to_pad(file_input_train):
    with open(file_word_list, 'r') as json_file:
        word_index = json.load(json_file)

    tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
    tokenizer.word_index = word_index

    # Tokenize và padding chuỗi văn bản
    input_sequences = tokenizer.texts_to_sequences(file_input_train)
    return pad_sequences(input_sequences, maxlen=number_of_input, padding='post', truncating='post')


def update_weights_on_incorrect_prediction(model, incorrect_sentence, correct_label):
    # Giả sử `convert_to_pad` là hàm tiền xử lý câu
    # print("cau:")
    # print(incorrect_sentence)
    # print(correct_label)
    incorrect_sentence_padded = convert_to_pad(incorrect_sentence)  
    
    # Bộ tối ưu và hàm mất mát cho bài toán hồi quy (Mean Squared Error)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()  # Dùng cho bài toán hồi quy
    
    correct_label = np.array([correct_label], dtype=np.float32)  # Đảm bảo correct_label là kiểu float
    best_loss = float('inf')  # Giá trị mất mát tốt nhất
    patience = ts.so_lan_loss_k_thay_doi  # Số lần cho phép mất mát không cải thiện
    wait = 0  # Bộ đếm số lần không cải thiện
    count = 0  # Bộ đếm tổng số lần lặp
    maxx=6000
    while True:
        with tf.GradientTape() as tape:
            logits = model(incorrect_sentence_padded, training=True)  # Tính toán đầu ra
            loss_value = loss_fn(correct_label, logits)  # Tính toán mất mát
        
        grads = tape.gradient(loss_value, model.trainable_weights)  # Tính gradient
        optimizer.apply_gradients(zip(grads, model.trainable_weights))  # Cập nhật trọng số
        
        # Kiểm tra nếu mất mát giảm
        if (loss_value < best_loss):
            best_loss = loss_value
            wait = 0  # Reset bộ đếm nếu có sự cải thiện
        else:
            wait += 1  # Tăng bộ đếm nếu không có sự cải thiện
        if (count>maxx):
            return  print(f"Đã dừng sau {count} vòng lặp với mất mát tốt nhất: {best_loss:.4f}")
            break
        count += 1
        
        # Dừng lại nếu không cải thiện sau `patience` lần hoặc đã vượt quá 1000 vòng lặp
        if wait >= patience :
            print(f"Đã dừng sau {count} vòng lặp với mất mát tốt nhất: {best_loss:.4f}")
            break

import tim_kiem_json as tkj
def load_model(k):
    name_mode=replace_space_with_underscore(k)
    new_model = create_model(len(tkj.search_name_lable(table_name=name_mode))+1)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #print(name_mode)
    new_model.load_weights(weight_model.format(name_mode))
    return new_model
def load_model_true_false(k):
    name_mode=replace_space_with_underscore(k)
    new_model = create_model(1+1)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    new_model.load_weights(weight_model.format(name_mode))
    return new_model
from itertools import combinations
def replace_positive(numbers):
    temp=[]
    # Lấy các chỉ mục của phần tử lớn hơn 0
    positive_indices = [i for i, num in enumerate(numbers) if num > 0]
    
    # Số lượng phần tử lớn hơn 0
    n = len(positive_indices)
    
    # Lặp qua số lượng phần tử thay thế (1 đến n)
    for k in range(1, n + 1):
        # Lấy tất cả các tổ hợp của k phần tử lớn hơn 0
        for positions in combinations(positive_indices, k):
            # Tạo bản sao của danh sách
            new_numbers = numbers[:]
            # Thay thế các vị trí được chọn thành 0
            for pos in positions:
                new_numbers[pos] = 0
            temp.append(new_numbers)
    return temp
import pyodbc
import data.tham_so as ts

def search_with_conditions_sqlserver( data ):

    conn = pyodbc.connect(
        f"DRIVER={{SQL Server}};SERVER={ts.server};DATABASE={ts.database};UID={ts.username};PWD={ts.password}"
    )
    cursor = conn.cursor()
    
    # Tạo truy vấn SQL để chỉ lấy cột `content`
    query = "SELECT content FROM answer WHERE {}_id = {} and {}_id = {} and {}_id = {} and {}_id = {} and {}_id = {} and {}_id = {} and {}_id = {} and {}_id = {} and {}_id = {};".format(ts.tables[0],data[0],ts.tables[1],data[1],ts.tables[2],data[2],ts.tables[3],data[3],ts.tables[4],data[4],ts.tables[5],data[5],ts.tables[6],data[6],ts.tables[7],data[7],ts.tables[8],data[8])
    
    try:
        # Thực thi truy vấn
        cursor.execute(query)
        results = cursor.fetchall()
        # Chỉ lấy giá trị của cột `content`
        content_results = [row[0] for row in results]
    except Exception as e:
        print(f"Lỗi khi thực thi truy vấn: {e}")
        content_results = []
    finally:
        # Đóng kết nối
        conn.close()
    
    return content_results
#print(search_with_conditions_sqlserver( [11, 0, 12, 0, 0, 0, 0, 0, 0] ))

def replace_space_with_underscore(input_string):
    """
    Chuyển đổi các khoảng trắng trong chuỗi thành dấu gạch dưới (_).
    
    Args:
        input_string (str): Chuỗi đầu vào.
    
    Returns:
        str: Chuỗi đã chuyển đổi.
    """
    return input_string.replace(" ", "_")