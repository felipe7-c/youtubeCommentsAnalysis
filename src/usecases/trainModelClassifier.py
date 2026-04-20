import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F

nltk.download('rslp')
nltk.download('punkt_tab')

class SimpleNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TrainModelClassifier:
    def __init__(self, data_path : str):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.stop_words = nltk.corpus.stopwords.words('portuguese')
        self.stemmer = nltk.stem.RSLPStemmer()        

    def preprocess_data(self, comment : str):

        #Minusculo
        comment = comment.lower()

        #Tokenização
        tokens = word_tokenize(comment, language='portuguese')

        #Remoção de stop words e stemming
        tokens = [word for word in tokens if word not in self.stop_words]

        tokens = [self.stemmer.stem(word) for word in tokens]

        tokens = ' '.join(tokens)

        return tokens
    
    def train_model(self):
        processed_comments = []
        labels = []

        label_map = {
            "positivo": 1,
            "negativo": 0,
            "neutro": 2
        }
        for index, row in self.df.iterrows():
            comment = row['comentarios']
            label = row['result']

            processed_comment = self.preprocess_data(comment)
            label = label_map[label]
            processed_comments.append(processed_comment)
            labels.append(label)

        #TF-IDF 
        vecotrizer = TfidfVectorizer(max_df=0.6, min_df = 3, max_features=5000)
        X = vecotrizer.fit_transform(processed_comments)
        X = X.toarray()
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 28)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        model = SimpleNN(input_size=X_train.shape[1], hidden_size = 100, num_classes = 3)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

        num_epochs = 50

        for epoch in range(num_epochs):
            model.train()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)

            accuracy = (predicted == y_test).sum().item() / len(y_test)
            print(f"Acurácia: {accuracy:.4f}")

        return model, vecotrizer, label_map



