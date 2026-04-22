# 🧠 NLP Comment Classification Pipeline

## 📌 Overview

Este projeto implementa um pipeline completo de Machine Learning para classificação de comentários, combinando coleta automatizada de dados, rotulação assistida por LLM e treinamento supervisionado.

A arquitetura foi projetada com foco em **modularidade, reprodutibilidade e escalabilidade**, seguindo boas práticas de sistemas de ML em produção.

---

## ⚙️ Arquitetura do Sistema

O pipeline é estruturado em etapas independentes:

Data Collection → Data Processing → LLM Labeling → Model Training → Inference

Cada etapa é desacoplada, permitindo evolução e substituição de componentes sem impacto no restante do sistema.

---

## 🧩 Componentes

### 🔹 Domain Layer

Camada responsável pela **orquestração das regras de negócio e fluxo do pipeline de ML**.

Atua como intermediária entre a orquestração principal (`main.py`) e os casos de uso, garantindo separação de responsabilidades e organização do fluxo.

**Responsabilidades:**
- Coordenar execução dos pipelines
- Encapsular lógica de alto nível
- Garantir consistência entre etapas do fluxo

**Componentes:**
- `collectDataDomain.py`
- `trainModelDomain.py`

---

### 🔹 Use Cases Layer

Camada responsável pela implementação das ações específicas do sistema:

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

Camada de orquestração do fluxo completo:

- Inicializa variáveis de ambiente
- Instancia camadas de domínio
- Executa pipeline de coleta e treinamento

**Fluxo:**
1. `CollectDataDomain` → coleta e rotula dados
2. `TrainModelDomain` → treina modelo supervisionado

**Output:**
- Assets/CommentsData.csv
- Models/

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

Camada de inferência exposta via API (FastAPI + Uvicorn):

- Carregamento dos artefatos treinados
- Pré-processamento consistente com treino
- Predição em tempo real via endpoint HTTP

---

## 🔄 Fluxo de Dados

main.py  
→ Domain Layer  
→ Use Cases  
→ Assets/ (dados)  
→ Models/ (modelo treinado)  
→ predict.py (API de inferência)

---

## 🧠 Arquitetura de Execução

### 🔹 Offline (Treinamento)

```bash
python src.main.py
```

- Coleta dados
- Rotula via LLM
- Treina modelo
- Salva artefatos

### 🔹 Online (Inferência)

```
uvicorn src.predict:app --reload
```

- Infere a partir do modelo salvo
