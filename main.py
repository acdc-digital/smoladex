import os
import sys
sys.path.append('./medex_project')
sys.path.append('./medex_project/data')

from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_sslify import SSLify
from config import Config
from transformers import RagTokenizer, RagTokenForGeneration, BartTokenizer, DPRQuestionEncoderTokenizer

from medex_project.preprocessing.clean_data import clean_data
from medex_project.preprocessing.tokenize_data import tokenize_data
from medex_project.preprocessing.split_data import split_data
from medex_project.model_parameters.medex_language_model import MedexLanguageModel
from medex_project.model_parameters.medex_training import train_medex_language_model
from medex_project.model_parameters.medex_evaluation import evaluate_model
from medex_project.model_parameters.medex_hyperparameters import Hyperparameters
from medex_project.utils.privacy_filter import PrivacyFilter
from medex_project.utils.interpretation import interpret_medical_data
from medex_project.utils.save_load_model import save_model, load_model
from model_parameters.val_data import val_data_loader

app = Flask(__name__)
app.config.from_object(Config)
sslify = SSLify(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    # Your code block here
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    # Add any other code that needs to be inside the app context

# Instantiate tokenizers
question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Instantiate RAG tokenizer
tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)

def main():
    # Load and preprocess data
    medical_corpus = os.path.join("medex_project", "data", "medical_corpus.txt")
    privacy_corpus = os.path.join("medex_project", "data", "privacy_corpus.txt")
    user_data_corpus = os.path.join("medex_project", "data", "user_data_corpus.txt")

    cleaned_data = clean_data(medical_corpus, privacy_corpus, user_data_corpus)
    cleaned_data_str = ' '.join(cleaned_data)  # Join the cleaned data into a single string
    tokenized_data = tokenize_data(cleaned_data_str)
    train_data, val_data, test_data = split_data(tokenized_data)

    # Instantiate the MedexLanguageModel
    vocab_size = 30522  # Update this value based on your dataset
    medex_language_model = MedexLanguageModel(vocab_size)

    # Load the train_data and val_data
    train_data = train_data_loader()
    val_data = val_data_loader()

    # Instantiate the Hyperparameters
    hyperparameters = Hyperparameters()

    # Train and evaluate the model_parameters
    trained_model = train_medex_language_model(medex_language_model, train_data, val_data, hyperparameters)
    evaluation_results = evaluate_model(trained_model, test_data)

    # Save the trained model_parameters
    model_path = os.path.join("medex_project", "medex_project/model_parameters", "trained_medex_language_model.pth")
    save_model(trained_model, model_path)

    # Load the trained model_parameters for use
    loaded_model = load_model(model_path)

    # Apply privacy filter and interpret medical data
    privacy_filter = PrivacyFilter()
    filtered_data = privacy_filter.apply_filter(user_data_corpus)
    interpreted_data = interpret_medical_data(loaded_model, filtered_data)

    print("Interpreted medical data:", interpreted_data)

if __name__ == "__main__":
    main()
    app.run(debug=True)