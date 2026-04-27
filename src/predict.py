from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sentence_transformers import SentenceTransformer


class CommentRequest(BaseModel):
    text: str

class PredictionService:
    def __init__(self):
        self.embedding_model, self.label_map, self.model = self.load_artifacts()

    def load_artifacts(self):
        # embedding model 
        embedding_model = SentenceTransformer("models/embedding_model")

        # label map
        with open("models/label_map.pkl", "rb") as f:
            label_map = pickle.load(f)

        # logistic regression model
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        return embedding_model, label_map, model

    def predict(self, text: str):

        X = self.embedding_model.encode([text])

        pred = self.model.predict(X)[0]

        # inversão do label map
        inv_map = {v: k for k, v in self.label_map.items()}

        return inv_map[pred]


app = FastAPI()
service = PredictionService()


@app.post("/predict")
def predict_endpoint(request: CommentRequest):
    try:
        return {"prediction": service.predict(request.text)}
    except Exception as e:
        return {"error": str(e)}