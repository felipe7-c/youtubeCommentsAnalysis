import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F

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




