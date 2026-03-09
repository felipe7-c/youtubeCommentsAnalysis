import os
from dotenv import load_dotenv
from src.usecases.collectCommentsUsecase import CollectCommentsUsecase

load_dotenv()
api_key = os.getenv("api_key")
channel_id = os.getenv("channel_id")
upload_playlist = os.getenv("uploads_playlist")

def main():

    #Instancia Usecase
    inst_collectData = CollectCommentsUsecase(api_key, channel_id, upload_playlist)

    #Collect Videos Comments
    comments = inst_collectData.collectCommentsVideos()

    return


if __name__ == "__main__":
    main()

