def max_sentence_length(file_path):
    max_length = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line into words and count the number of words
            word_count = len(line.split())
            # Update max_length if the current line is longer
            if word_count > max_length:
                max_length = word_count
    return max_length

# Usage
file_path = 'data_train\input_train\content_question-test.ta'
longest_sentence_length = max_sentence_length(file_path)
print("The longest sentence has", longest_sentence_length, "words.")
