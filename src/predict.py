import torch
import pickle
import nltk
from nltk.tokenize import word_tokenize
from src.usecases.trainModelClassifier import SimpleNN

nltk.download('punkt_tab')
nltk.download('rslp')

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

input_size = len(vectorizer.get_feature_names_out())

model = SimpleNN(input_size=input_size, hidden_size=100, num_classes=3)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text, language='portuguese')

    stop_words = nltk.corpus.stopwords.words('portuguese')
    stemmer = nltk.stem.RSLPStemmer()

    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]

    return ' '.join(tokens)


def predict(text):
    processed = preprocess(text)

    X = vectorizer.transform([processed]).toarray()
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X)
        _, pred = torch.max(output, 1)

    inv_map = {v: k for k, v in label_map.items()}

    return inv_map[pred.item()]


if __name__ == "__main__":
    texto = "Trump é um péssimo presidente!"
    print(predict(texto))