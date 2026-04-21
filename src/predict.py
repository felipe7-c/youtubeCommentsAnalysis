import torch
import pickle
import nltk
from nltk.tokenize import word_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
from src.usecases.trainModelClassifier import SimpleNN

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

class CommentRequest(BaseModel):
    text: str

class PredictionService:
    def __init__(self):
        self.vectorizer, self.label_map, self.model = self.load_artifacts()
        self.stop_words = nltk.corpus.stopwords.words('portuguese')
        self.stemmer = nltk.stem.RSLPStemmer()

    def load_artifacts(self):
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open("models/label_map.pkl", "rb") as f:
            label_map = pickle.load(f)

        input_size = len(vectorizer.get_feature_names_out())

        model = SimpleNN(input_size=input_size, hidden_size=100, num_classes=3)
        model.load_state_dict(torch.load("models/model.pth"))
        model.eval()

        return vectorizer, label_map, model
    
    def preprocess(self, text):
        text = text.lower()
        tokens = word_tokenize(text, language='portuguese')

        tokens = [w for w in tokens if w not in self.stop_words]
        tokens = [self.stemmer.stem(w) for w in tokens]

        return ' '.join(tokens)


    def predict(self, text):
        processed = self.preprocess(text)

        X = self.vectorizer.transform([processed]).toarray()
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(X)
            _, pred = torch.max(output, 1)

        inv_map = {v: k for k, v in self.label_map.items()}

        return inv_map[pred.item()]

app = FastAPI()

service = PredictionService()

@app.post("/predict")
def predict_endpoint(request: CommentRequest):
    try: 
        return {"prediction": service.predict(request.text)}
    except Exception as e:
        return {"error": str(e)}