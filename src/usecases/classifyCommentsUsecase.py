from groq import Groq

class ClassifyCommentsUsecase:
    def __init__(self, grok_api_key: str, comments: list[str], model_name : str):
        self.grok_api_key = grok_api_key
        self.comments = comments
        self.client = Groq(api_key = self.grok_api_key)
        self.model_name = model_name
        self.temperature = 0
        self.max_completion_tokens = 10
    def classify(self):

        results = []

        for comment_number in range(0, len(self.comments), 3):

            comments = self.comments[comment_number: comment_number + 3]

            comments = [comment.strip() for comment in comments if comment.strip() != ""]
                
            if not comments:
                continue

            prompt = f"""
            Classifique os comentários político como:
            - positivo
            - negativo
            - neutro

            classifique conforme a ordem dos comentários abaixo com apenas uma palavra, 
            um exemplo de resposta seria: "positivo, negativo, neutro, negativo, neutro" 
            Atenção: não utilize abreviações apenas as palavras completas citadas acima.

            Comentários: {"; ".join(f'"{comment}"' for comment in comments)}
            """

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,  
                max_completion_tokens=self.max_completion_tokens
            )

            label = completion.choices[0].message.content.strip().lower()
            array_label = label.split(",")
            results.extend(array_label)

        return results

