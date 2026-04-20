from src.usecases.trainModelClassifier import TrainModelClassifier
from pathlib import Path
import torch
import pickle

project_root = Path(__file__).parent.parent.resolve()  

csv_path = project_root / "assets" / "comments_data.csv"
model_path = project_root / "models"
model_path.mkdir(parents=True, exist_ok=True)

def main():

    inst_train_model = TrainModelClassifier(csv_path)

    model, vectorizer, label_map = inst_train_model.train_model()

    torch.save(model.state_dict(), model_path / "model.pth")

    with open(model_path / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(model_path / "label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)

    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    main()