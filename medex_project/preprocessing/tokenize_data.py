import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

def tokenize_data(cleaned_data):
    sentences = sent_tokenize(cleaned_data)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_sentences

def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        tokenized_sentences = tokenize_data(text)
    return tokenized_sentences

def save_tokenized_data(tokenized_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in tokenized_data:
            file.write(' '.join(sentence) + '\n')

def tokenize_data_files():
    input_files = [
        "medex_project/data/medical_corpus.txt",
        "medex_project/data/privacy_corpus.txt",
        "medex_project/data/user_data_corpus.txt"
    ]
    output_files = [
        "medex_project/data/tokenized_medical_corpus.txt",
        "medex_project/data/tokenized_privacy_corpus.txt",
        "medex_project/data/tokenized_user_data_corpus.txt"
    ]

    for input_file, output_file in zip(input_files, output_files):
        tokenized_data = tokenize_file(input_file)
        save_tokenized_data(tokenized_data, output_file)

if __name__ == "__main__":
    tokenize_data_files()