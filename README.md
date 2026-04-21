# 🧠 NLP Comment Classification Pipeline

## 📌 Overview

Este projeto implementa um pipeline completo de Machine Learning para classificação de comentários, combinando coleta automatizada de dados, rotulação assistida por LLM e treinamento supervisionado.

A arquitetura foi projetada com foco em **modularidade, reprodutibilidade e escalabilidade**, seguindo boas práticas de sistemas de ML em produção.

---

## ⚙️ Arquitetura do Sistema

O pipeline é estruturado em etapas independentes:
- Data Collection → Data Processing → LLM Labeling → Model Training → Inference


Cada etapa é desacoplada, permitindo evolução e substituição de componentes sem impacto no restante do sistema.

---

## 🧩 Componentes

### 🔹 Use Cases Layer

Camada responsável pelos fluxos de negócio:

- Coleta de comentários via API (YouTube)
- Classificação de texto
- Geração de dataset rotulado utilizando LLM (data-centric approach)

---

### 🔹 Data Processing Layer

Responsável pelo pré-processamento dos dados textuais:

- Limpeza (remoção de emojis, caracteres especiais)
- Normalização (case folding)
- Preparação para vetorização

---

### 🔹 EDA & Analysis (Notebooks)

Ambiente dedicado à análise exploratória:

- Distribuição de classes
- Identificação de desbalanceamento
- Avaliação da qualidade dos dados rotulados

---

### 🔹 Data Pipeline (`main.py`)

Orquestração do fluxo de dados:

- Coleta comentários da API
- Integração com LLM para rotulação automática
- Persistência dos dados processados

**Output:**
- Assets/CommentsData.csv


---

### 🔹 Training Pipeline (`train.py`)

Pipeline de treinamento supervisionado:

- Vetorização de texto (TF-IDF)
- Treinamento do modelo de classificação
- Serialização dos artefatos

**Outputs:**
- Models/model.pth
- Models/vectorizer.pkl
- Models/labelmap.pkl

---

### 🔹 Model Artifacts

Separação clara entre dados e modelo:

- **Assets/**
  - Dataset processado e rotulado
- **Models/**
  - Modelo treinado
  - Vetorizador
  - Mapeamento de labels

Essa estrutura garante **reprodutibilidade e portabilidade do modelo**.

---

### 🔹 Inference Layer (`predict.py`)

Camada de inferência:

- Carregamento dos artefatos
- Transformação dos dados de entrada
- Predição em novos comentários

---

## 🔄 Fluxo de Dados
- main.py → gera dataset → Assets/
- train.py → treina modelo → Models/
- predict.py → consome modelo → predições
