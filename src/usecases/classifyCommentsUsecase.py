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

        for comment in self.comments:

            if comment.strip() == "":
                continue

            prompt = f"""
            Classifique o comentário político como:
            - positivo
            - negativo
            - neutro

            Responda apenas com uma palavra.

            Comentário: "{comment}"
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
            results.append(label)

        return results

