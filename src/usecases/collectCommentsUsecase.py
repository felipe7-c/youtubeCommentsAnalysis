from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class CollectCommentsUsecase:

    def __init__(self, api_key, channel_id, upload_playlist):
        self.api_key = api_key
        self.channel_id = channel_id
        self.upload_playlist = upload_playlist
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    def collectVideoIds(self, max_videos=10):

        videos = []
        next_page = None

        while True:

            request = self.youtube.playlistItems().list(
                part="snippet",
                playlistId=self.upload_playlist,
                maxResults=50,
                pageToken=next_page
            )

            response = request.execute()

            for item in response["items"]:
                video_id = item["snippet"]["resourceId"]["videoId"]
                videos.append(video_id)

                if len(videos) >= max_videos:
                    return videos

            next_page = response.get("nextPageToken")

            if not next_page:
                break

        return videos


    def collectCommentsVideos(self, max_videos=10, max_comments_per_video=20):

        comments = []

        videos = self.collectVideoIds(max_videos)

        for video_id in videos:

            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_comments_per_video
            )

            try:
                response = request.execute()

                for item in response["items"]:
                    text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(text)
            except HttpError as e:

                if "commentsDisabled" in str(e):
                    print(f"Comentários desativados no vídeo {video_id}")
                    continue
                else:
                    raise e

        return comments