from groq import Groq
from tqdm import tqdm
import ast

class ClassifyCommentsUsecase:
    def __init__(self, grok_api_key: str, comments: list[str], model_name : str):
        self.grok_api_key = grok_api_key
        self.comments = comments
        self.client = Groq(api_key = self.grok_api_key)
        self.model_name = model_name
        self.temperature = 0
        self.max_completion_tokens = 50
    def classify(self):

        results = []
        comments_case = []

        for comment_number in tqdm( range(0, len(self.comments), 5)):

            comments = self.comments[comment_number: comment_number + 5]

            comments = [comment.strip() for comment in comments if comment.strip() != ""]
                
            if not comments:
                continue

            prompt = f"""
                Classifique cada comentário político como:
                - positivo
                - negativo
                - neutro

                Responda SOMENTE em uma lista Python válida, com o mesmo tamanho dos comentários.

                Exemplo:
                ["positivo", "negativo", "neutro", "negativo", "neutro"]

                Comentários:
                {chr(10).join(comments)}
                """
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens
                )

                label = completion.choices[0].message.content.strip().lower()
                try:
                    array_label = ast.literal_eval(label)
                except Exception as e:
                    print("Erro ao fazer parse:", label)
                    continue

                if len(array_label) != len(comments):
                    min_len = min(len(array_label), len(comments))

                    array_label = array_label[:min_len]
                    comments = comments[:min_len]

                results.extend(array_label)
                comments_case.extend(comments)

            except Exception as e:
                print(f"Erro ao classificar comentários: {e}")
                continue

        return results, comments_case

