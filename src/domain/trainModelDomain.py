from src.usecases.trainModelClassifier import TrainModelClassifier
import torch
import pickle
from pathlib import Path

class TrainModelDomain:
    def __init__(self, csv_path : str):
        self.csv_path = csv_path
        project_root = Path(__file__).parent.parent.resolve()  

        csv_path = project_root / "assets" / "comments_data.csv"
        self.model_path = project_root / "models"
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train_model(self):

        inst_train_model = TrainModelClassifier(self.csv_path)

        model, vectorizer, label_map = inst_train_model.train_model()

        torch.save(model.state_dict(), self.model_path / "model.pth")

        with open(self.model_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        with open(self.model_path / "label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)

        print("Modelo salvo com sucesso!")

        return