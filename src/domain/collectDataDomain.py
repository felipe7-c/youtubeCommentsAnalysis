from src.usecases.collectCommentsUsecase import CollectCommentsUsecase
from src.usecases.classifyCommentsUsecase import ClassifyCommentsUsecase
from src.data.cleaning import DataProcessing
import pandas as pd
from pathlib import Path
import pandas as pd
import os

class CollectDataDomain:
    
    def __init__(self, api_key, channel_id, upload_playlist, grok_api_key, model_name):
        self.api_key = api_key
        self.channel_id = channel_id
        self.upload_playlist = upload_playlist
        self.grok_api_key = grok_api_key
        self.model_name = model_name

    def getTreatedComments(self):
        #Instancia Usecase
        inst_collectData = CollectCommentsUsecase(self.api_key, self.channel_id, self.upload_playlist)

        #Collect Videos Comments
        comments = inst_collectData.collectCommentsVideos()

        inst_comments_treat = DataProcessing(comments)
        
        treated_data = inst_comments_treat.cleaningData()

        return treated_data

    def collectAndClassifyComments(self):

        project_root = Path(__file__).parent.parent.resolve()  

        csv_path = project_root / "assets" / "comments_data.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        comments_treated = self.getTreatedComments()

        results_comments, comments_case = ClassifyCommentsUsecase(grok_api_key=self.grok_api_key, comments=comments_treated, model_name=self.model_name).classify()

        df = pd.DataFrame({"comentarios" : comments_case, "result" : results_comments})

        # Salva CSV
        if os.path.exists(csv_path):
            df_before = pd.read_csv(csv_path)
            df = pd.concat([df_before, df], ignore_index = True)

        df.to_csv(csv_path, index=False)
        print(f"CSV salvo em: {csv_path}")

        return