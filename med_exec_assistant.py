from transformers import RagTokenForGeneration, RagTokenizer

# Instantiate RAG model_parameters and tokenizer
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
