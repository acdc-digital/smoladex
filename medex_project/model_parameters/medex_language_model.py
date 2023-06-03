import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer

class MedexLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(MedexLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

def train_medex_language_model(train_data, val_data, epochs, batch_size, learning_rate, device):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.5

    model = MedexLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        hidden = None
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs, hidden)
                val_loss += criterion(outputs.view(-1, vocab_size), targets.view(-1)).item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}")

    save_model(model, "medex_project/model_parameters/medex_language_model.pth")

def load_medex_language_model(device):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.5

    model = MedexLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
    model = load_model(model, "medex_project/model_parameters/medex_language_model.pth")
    return model