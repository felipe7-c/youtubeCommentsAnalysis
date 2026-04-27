import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('rslp')
nltk.download('punkt_tab')

class TrainModelClassifier:
    def __init__(self, data_path : str, data_comp_path : str):
        self.data_path = data_path
        if data_comp_path is None:
            self.df = pd.read_csv(self.data_path)
        else:
            df1 = pd.read_csv(self.data_path)
            df2 = pd.read_csv(data_comp_path, encoding="latin-1")
            self.df = pd.concat([df1, df2], ignore_index=True)

        self.df["comentarios"] = self.df["comentarios"].fillna("").astype(str)
        self.df["result"] = self.df["result"].fillna("").astype(str)
        self.df["result"] = self.df["result"].str.strip().str.lower()

        self.df = self.df[self.df["result"].isin(["positivo", "negativo", "neutro"])]

        print(self.df["result"].value_counts())
        
        self.stop_words = nltk.corpus.stopwords.words('portuguese')
        self.stemmer = nltk.stem.RSLPStemmer()        

    def preprocess_data(self, comment : str):

        if not isinstance(comment, str):
            return ""

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

        #Balanceamento de classes: amostragem aleatória de 1000 comentários
        df_neg = self.df[self.df["result"] == "negativo"].sample(1000)
        df_other = self.df[self.df["result"] != "negativo"]

        self.df = pd.concat([df_neg, df_other])

        for _, row in self.df.iterrows():
            comment = row['comentarios']
            label = row['result']

            processed_comment = self.preprocess_data(comment)
            processed_comments.append(processed_comment)
            labels.append(label_map[f'{label.strip()}'])
        
        vectorizer = TfidfVectorizer(
            max_df=0.6,
            min_df=3,
            max_features=5000,
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(processed_comments)
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=28, stratify = y
        )

        model = LogisticRegression(
            max_iter = 1000,
            random_state = 42,
            class_weight = "balanced"   
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        return model, vectorizer, label_map