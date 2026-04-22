import os
from dotenv import load_dotenv
from pathlib import Path
from src.domain.collectDataDomain import CollectDataDomain
from src.domain.trainModelDomain import TrainModelDomain

load_dotenv()
api_key = os.getenv("api_key")
channel_id = os.getenv("channel_id")
upload_playlist = os.getenv("uploads_playlist")
grok_api_key = os.getenv("grok_api_key")
model_name = os.getenv("model_name")

def main():

    project_root = Path(__file__).parent.parent.resolve()  

    csv_path = project_root / "assets" / "comments_data.csv"
    model_path = project_root / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    collect_data_domain = CollectDataDomain(api_key, channel_id, upload_playlist, grok_api_key, model_name)
    train_model_domain = TrainModelDomain(csv_path)

    #Coleta os dados rotulados
    collect_data_domain.collectAndClassifyComments()

    #Treina o modelo e salva
    train_model_domain.train_model()

    return

if __name__ == "__main__":
    main()