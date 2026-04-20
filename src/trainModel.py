from src.usecases.trainModelClassifier import TrainModelClassifier
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()  

csv_path = project_root / "assets" / "comments_data.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

# Treina modelo de classificação
inst_train_model = TrainModelClassifier(csv_path)
inst_train_model.train_model()