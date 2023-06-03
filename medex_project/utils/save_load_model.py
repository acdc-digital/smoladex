import torch
from medex_project.model_parameters.medex_language_model import MedexLanguageModel

def save_model(model: MedexLanguageModel, model_path: str):
    torch.save(model.state_dict(), model_path)

def load_model(model_path: str, device: torch.device) -> MedexLanguageModel:
    model = MedexLanguageModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model