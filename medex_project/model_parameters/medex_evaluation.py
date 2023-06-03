import torch
from torch.utils.data import DataLoader
from medex_project.model_parameters.medex_language_model import MedexLanguageModel
from medex_project.utils.save_load_model import load_model

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            input_data, target_data = batch
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            output = model(input_data)
            loss = criterion(output.view(-1, model.vocab_size), target_data.view(-1))
            total_loss += loss.item()

    return total_loss / len(test_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "medex_project/model_parameters/saved_model.pth"
    test_data_path = "medex_project/data/test_data.txt"

    model = MedexLanguageModel()
    model = load_model(model, model_path, device)

    test_dataset = MedexDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()