import re

class DataProcessing:
    def __init__(self, comments: list[str]):
        self.comments = comments

    def cleaningData(self):

        cleaned_comments = []

        for comment in self.comments:
            cleaned_comment = self.transformingData(comment)
            if cleaned_comment != "":
                cleaned_comments.append(cleaned_comment)
        
        return cleaned_comments
    
    def transformingData(self, comment: str):

        comment = comment.lower()

        # remove links
        comment = re.sub(r"http\S+|www\S+", "", comment)

        # remove menções
        comment = re.sub(r"@\w+", "", comment)

        # remove hashtags
        comment = re.sub(r"#\w+", "", comment)

        # remove emojis
        comment = re.sub(r"[^\w\sà-úÀ-Ú]", "", comment)

        # remove números
        comment = re.sub(r"\d+", "", comment)

        # remove espaços duplicados
        comment = re.sub(r"\s+", " ", comment)

        return comment.strip()

