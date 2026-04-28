from src.usecases.trainModelClassifier import TrainModelClassifier
import pickle
from pathlib import Path
import os

class TrainModelDomain:
    def __init__(self, csv_path: str, project_root: str):
        self.csv_path = csv_path
        self.csv_comp_path = None
        path_comp = project_root/ "assets" / "comments_data_comp.csv"

        if os.path.exists(path_comp):
            self.csv_comp_path = path_comp

        self.project_root = project_root

        csv_path = project_root / "assets" / "comments_data.csv"
        self.model_path = project_root / "models"
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train_model(self):

        inst_train_model = TrainModelClassifier(self.csv_path, self.csv_comp_path)

        model, embedding_model, label_map = inst_train_model.train_model()

        # Salvar modelo de classificação (sklearn)
        with open(self.model_path / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Salvar label map
        with open(self.model_path / "label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)

        print("Modelo salvo com sucesso!")

        return