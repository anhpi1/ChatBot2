import os
import itertools
from transformers import pipeline
from collections import Counter
import pyodbc
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Disable TensorFlow OneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Input data
input_data = []
with open('input.txt', 'r') as f:
    input_data = [line.strip() for line in f]

# Database connection
topics = ['question_type', 'question_intent', 'concept1', 'concept2', 'structure', 'performance_metric']
nhan_o_dang_chu = []

conn = pyodbc.connect(
    r'DRIVER={SQL Server};'
    r'SERVER=DESKTOP-1MU0IU3\SQLEXPRESS;'
    r'DATABASE=comparator;'
    r'UID=;'
    r'PWD='
)
cursor = conn.cursor()
for topic in topics:
    cursor.execute("SELECT content FROM {}".format(topic))
    results = cursor.fetchall()
    nhan_o_dang_chu.append([row[0] for row in results])

cursor.close()

# Classification function
def classify_text(args):
    model_name, input_text, candidate_labels = args
    classifier = pipeline("zero-shot-classification", model=model_name, truncation=True)
    du_doan = []
    for line in input_text:
        result = classifier(line, candidate_labels=candidate_labels, max_length=32, multi_label=False)
        du_doan.append(result['labels'][0])
    return du_doan

# Models for classification
models = [
    "tasksource/deberta-small-long-nli",
    "tasksource/deberta-base-long-nli",
]

# Parallel classification using ThreadPoolExecutor
du_doan_dang_chu = []
with ThreadPoolExecutor() as executor:
    tasks = [
        (models[i], input_data, nhan_o_dang_chu[i]) for i in range(len(models))
    ]
    results = executor.map(classify_text, tasks)
    du_doan_dang_chu.extend(results)

# Convert predictions to database IDs
du_doan_dang_so = []
cursor = conn.cursor()
for i, predictions in enumerate(du_doan_dang_chu):
    temp = []
    for element in predictions:
        cursor.execute("SELECT id FROM {} WHERE CONVERT(VARCHAR(MAX), content) = '{}';".format(topics[i], element))
        result = cursor.fetchall()
        if result:
            temp.append(result)
        else:
            temp.append([(0,)])  # Giá trị mặc định nếu không tìm thấy
    du_doan_dang_so.append(temp)

# Flatten and normalize
def fun(temp):
    return list(itertools.chain(*[sublist[0] for sublist in temp if sublist]))

du_doan_dang_so = [fun(line) for line in du_doan_dang_so]

# Handle inhomogeneous lists for transpose
max_len = max(len(row) for row in du_doan_dang_so)
du_doan_dang_so = [row + [None] * (max_len - len(row)) for row in du_doan_dang_so]

# Transpose results
du_doan_dang_so_T = np.transpose(du_doan_dang_so)

# Count most common predictions
final_predictions = []
for line in du_doan_dang_so_T:
    count = Counter(line)
    most_common = count.most_common(1)
    final_predictions.append(most_common)

print(final_predictions)
conn.close()
