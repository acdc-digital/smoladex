import re
from typing import List

def clean_text(text: str) -> str:
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_data(medical_corpus: List[str], privacy_corpus: List[str], user_data_corpus: List[str]) -> List[str]:
    cleaned_data = medical_corpus + privacy_corpus + user_data_corpus
    cleaned_data = [clean_text(text) for text in cleaned_data]
    return cleaned_data