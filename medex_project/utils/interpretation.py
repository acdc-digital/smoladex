import re
from typing import List, Tuple

from medex_project.data.medical_corpus import medicalTerms, medicalAbbreviations

def parse_medical_language(text: str) -> List[Tuple[str, str]]:
    parsed_terms = []
    words = text.split()

    for word in words:
        if word in medicalTerms:
            parsed_terms.append((word, medicalTerms[word]))
        elif word in medicalAbbreviations:
            parsed_terms.append((word, medicalAbbreviations[word]))

    return parsed_terms

def interpret_medical_data(user_data: str, medical_data: str) -> str:
    interpretation = ""

    for term, definition in parse_medical_language(user_data):
        interpretation += f"{term}: {definition}\n"

    return interpretation.strip()

def get_medical_data_summary(medical_data: str) -> str:
    # You can update this function to extract and format the summary from the medical_data string
    summary = medical_data
    return summary.strip()