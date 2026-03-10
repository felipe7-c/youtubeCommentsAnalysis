import os
from dotenv import load_dotenv
from src.usecases.collectCommentsUsecase import CollectCommentsUsecase
from src.data.cleaning import DataProcessing
import pandas as pd
from pathlib import Path
import pandas as pd

load_dotenv()
api_key = os.getenv("api_key")
channel_id = os.getenv("channel_id")
upload_playlist = os.getenv("uploads_playlist")

def getTreatedComments(api_key, channel_id, upload_playlist):
    #Instancia Usecase
    inst_collectData = CollectCommentsUsecase(api_key, channel_id, upload_playlist)

    #Collect Videos Comments
    comments = inst_collectData.collectCommentsVideos()

    inst_comments_treat = DataProcessing(comments)
    
    treated_data = inst_comments_treat.cleaningData()

    return treated_data

def main():

    comments_treated = getTreatedComments(api_key, channel_id, upload_playlist)

    df = pd.DataFrame(comments_treated, columns=["comment"])

    project_root = Path(__file__).parent.parent.resolve()  # src/.. = raiz

    csv_path = project_root / "assets" / "comments_data.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Salva CSV
    df = pd.DataFrame(comments_treated, columns=["comment"])
    df.to_csv(csv_path, index=False)

    print(f"CSV salvo em: {csv_path}")
    
    return


if __name__ == "__main__":
    main()

