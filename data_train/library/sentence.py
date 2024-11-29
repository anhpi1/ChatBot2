from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import json
import data_train.library.train_TNN as TNN
import data_train.library.module_DST as DST
import copy
import os
import tensorflow as tf
# Sử dụng tất cả lõi CPU có sẵn
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

def sentencess(input_sentence, dst):
    # Tạo bản sao lưu của DST ban đầu
    dst_temp = copy.deepcopy(dst)

    # Khởi tạo tham số
    number_of_input = 0
    file_word_list = ''
    num_words_list = 0
    number_of_outputs = 0
    number_of_model = 0
    number_of_copies_model = 0
    Bt = []
    Ut = []
    dst = DST.DST_block()
    output_train = ''
    weight_model = ''
    # Tải tham số
    with open("parameter.ta", "r") as file:
        lines = file.readlines()
    for line in lines:
        # Bỏ qua dòng trống
        if not line.strip():
            continue
        key, value = line.split(" = ")
        key, value = key.strip(), value.strip()
        if key == "number_of_input" and value.isdigit():
            number_of_input = int(value)
        elif key == "num_words_list" and value.isdigit():
            num_words_list = int(value)
        elif key == "number_of_copies_model" and value.isdigit():
            number_of_copies_model = int(value)
        elif key == "number_of_model" and value.isdigit():
            number_of_model = int(value)
        elif key == "file_word_list":
            file_word_list = value.strip("'")
        elif key == "output_train":
            output_train = value.strip("'")
        elif key == "weight_model":
            weight_model = value.strip("'")

    # Tải word index
    try:
        with open(file_word_list, 'r') as json_file:
            word_index = json.load(json_file)
    except FileNotFoundError:
        print(f"Không tìm thấy tệp danh sách từ {file_word_list}.")
        return None

    tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
    tokenizer.word_index = word_index

    # Xử lý từng mô hình
    for name_mode in range(number_of_model):
        models = []
        temp = []

        # Tải các mô hình với trọng số tương ứng
        for i in range(0,number_of_copies_model):
            file_output_train = output_train.format(name_mode)
            try:
                with open(file_output_train, "r") as file:
                    numbers = file.readlines()
                number_of_outputs = max(int(number.strip()) for number in numbers) + 1
            except FileNotFoundError:
                print(f"Không tìm thấy tệp đầu ra train {file_output_train}.")
                continue

            new_model = TNN.create_model(number_of_outputs, number_of_input, num_words_list)
            new_model.load_weights(weight_model.format(name_mode,i))
            models.append(new_model)

        # Mã hóa và padding câu
        sequence = tokenizer.texts_to_sequences([input_sentence])
        padded_sequence = pad_sequences(sequence, maxlen=number_of_input)
        Ut = np.array(padded_sequence)

        # Dự đoán và cập nhật trọng số
        for model in models:
            predictions = model.predict(Ut, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            temp.append(predicted_class)

        values, counts = np.unique(temp, return_counts=True)
        most_frequent = values[np.argmax(counts)]
        Bt.append(most_frequent)
        del models  # Xóa các mô hình khỏi bộ nhớ sau khi sử dụng

    # Cập nhật DST
    dst.update(Bt=Bt, Ut=Ut, DST_history=dst_temp)
    return dst