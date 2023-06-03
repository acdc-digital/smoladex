import os
import random
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def split_data(data, val_size=0.2, test_size=0.2, random_state=None):
    train_data, temp_data = train_test_split(data, test_size=val_size + test_size, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size), random_state=random_state)
    return train_data, val_data, test_data

def save_data(file_path, data):
    with open(file_path, 'w') as file:
        file.writelines(data)

def main():
    medical_corpus_path = "medex_project/data/medical_corpus.txt"
    privacy_corpus_path = "medex_project/data/privacy_corpus.txt"
    user_data_corpus_path = "medex_project/data/user_data_corpus.txt"

    medical_data = load_data(medical_corpus_path)
    privacy_data = load_data(privacy_corpus_path)
    user_data = load_data(user_data_corpus_path)

    train_medical_data, val_medical_data, test_medical_data = split_data(medical_data)
    train_privacy_data, val_privacy_data, test_privacy_data = split_data(privacy_data)
    train_user_data, val_user_data, test_user_data = split_data(user_data)

    save_data("medex_project/data/train_medical_data.txt", train_medical_data)
    save_data("medex_project/data/test_medical_data.txt", test_medical_data)
    save_data("medex_project/data/val_medical_data.txt", val_medical_data)
    save_data("medex_project/data/train_privacy_data.txt", train_privacy_data)
    save_data("medex_project/data/test_privacy_data.txt", test_privacy_data)
    save_data("medex_project/data/val_privacy_data.txt", val_privacy_data)
    save_data("medex_project/data/train_user_data.txt", train_user_data)
    save_data("medex_project/data/test_user_data.txt", test_user_data)
    save_data("medex_project/data/val_user_data.txt", val_user_data)

if __name__ == "__main__":
    main()