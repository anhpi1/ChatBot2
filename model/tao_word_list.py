from tensorflow.keras.preprocessing.text import Tokenizer
import json
from data.tham_so import num_words_list,file_word_list

x=[]
with open('model\\data\\json\\data_column.json', 'r', encoding='utf-8') as data_file:
    x =json.load(data_file)


# tạo word list
tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
with open(file_word_list, 'w') as json_file:
    json.dump(word_index, json_file)

