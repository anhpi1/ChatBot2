from sentence_transformers import CrossEncoder
import concurrent.futures

# Load the CrossEncoder model
model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')

# Define pairs of sentences for classification
sentence_pairs = [
    ("I love programming in Python", "Python is my favorite programming language"),
    ("The stock market is volatile today", "The financial market shows fluctuations"),
    ("Climate change is a serious issue", "We must take action on environmental issues")
]

# Function to perform the classification
def classify_pair(pair):
    return model.predict([pair])[0]  # Use model.predict() for inference on pairs

# Use ThreadPoolExecutor to perform predictions in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(classify_pair, sentence_pairs))

# Print results (scores)
for result in results:
    print(result)