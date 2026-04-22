FROM python:3.10-slim

WORKDIR /app

# Copia tudo
COPY . .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Porta da API
EXPOSE 8000

# Comando para rodar API
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]