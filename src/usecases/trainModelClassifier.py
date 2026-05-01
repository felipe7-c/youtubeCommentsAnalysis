import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

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

    def preprocess_data(self, comment: str):
        if not isinstance(comment, str):
            return ""
        return comment.lower().strip()      
    
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

        processed_comments = self.df['comentarios'].apply(self.preprocess_data)
        labels = self.df['result'].apply(lambda x : label_map[f'{x.strip()}'])
        
        model_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        X = model_emb.encode(processed_comments.to_list())
        y = labels

        # Adicionar índices para rastrear os comentários originais
        indices = pd.Series(range(len(self.df)))

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices, test_size=0.2, random_state=28, stratify = y
        )

        model = LogisticRegression(
            max_iter = 1000,
            random_state = 42,
            class_weight = "balanced"   
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        reverse_label_map = {v: k for k, v in label_map.items()}
        
        y_proba = model.predict_proba(X_test)
        confidence = y_proba.max(axis=1)
        
        test_data = pd.DataFrame({
            'comentarios': self.df.iloc[idx_test.values]['comentarios'].values,
            'label_verdadeiro': [reverse_label_map[label] for label in y_test.values],
            'label_predito': [reverse_label_map[label] for label in y_pred],
            'confidence': confidence
        })

        return model, model_emb, label_map, test_data, X_test, y_test