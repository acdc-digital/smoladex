import torch
from torch.utils.data import DataLoader
from medex_project.model_parameters.medex_language_model import MedexLanguageModel
from medex_project.model_parameters.medex_hyperparameters import Hyperparameters


def train_medex_language_model(medex_language_model: MedexLanguageModel, train_data, val_data, hyperparameters: Hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medex_language_model.to(device)

    train_loader = DataLoader(train_data, batch_size=hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hyperparameters.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(medex_language_model.parameters(), lr=hyperparameters.learning_rate)

    for epoch in range(hyperparameters.epochs):
        medex_language_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_data, labels = batch
            input_data, labels = input_data.to(device), labels.to(device)

            outputs = medex_language_model(input_data)
            loss = medex_language_model.compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        medex_language_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_data, labels = batch
                input_data, labels = input_data.to(device), labels.to(device)

                outputs = medex_language_model(input_data)
                loss = medex_language_model.compute_loss(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}")

    return medex_language_model