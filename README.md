# Financial RAG Analyzer

> Systeme de questions-reponses sur rapports financiers annuels.
> Pipeline RAG complet : PDF → Chunking → Embeddings → Retrieval → LLM → Reponse citee.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Mistral](https://img.shields.io/badge/Ollama-Mistral_7B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---
## 🌐 Application en ligne

**Tester l'app :** https://financial-rag-analyzer-qgfb2n3bbdm74y8uyde8qu.streamlit.app/

---

## Objectif

Permettre a un analyste financier de poser des questions en langage naturel
sur des rapports annuels PDF et obtenir des reponses precises avec citations
des sources exactes (document + numero de page).

---

## Exemple

```
Question : Quelle est la strategie de BNP Paribas pour 2025 ?

Reponse  : La strategie de BNP Paribas est de poursuivre ses activites
           au plus pres des besoins de ses clients tout en contribuant
           a une economie durable [Source 1 — page 3].
           La banque a obtenu un score de 4,4/5 pour sa durabilite
           financiere pour la troisieme annee consecutive [Source 3 — page 17].
```

---

## Pipeline RAG

```
PDF → pdfplumber extraction
    → RecursiveCharacterTextSplitter (500 chars, overlap 50)
    → Sentence-Transformers all-MiniLM-L6-v2 (384 dims)
    → Cosine similarity retrieval (numpy)
    → Prompt anti-hallucination
    → Ollama Mistral 7B
    → Reponse avec citations [Source N — page X]
```

---

## Documents analyses

| Document | Pages | Mots | Langue | Chunks |
|---|---|---|---|---|
| BNP Paribas — Rapport Integre 2023 | 43 | 34 678 | FR | 517 |
| TotalEnergies — URD 2023 | 674 | 450 595 | EN | 6 892 |
| **Total** | **717** | **485 273** | | **7 409** |

---

## Resultats

| Question | Score max | Score moyen | Statut |
|---|---|---|---|
| Strategie BNP 2025 | 0.7559 | 0.6854 | OK |
| TotalEnergies renewables | 0.7667 | 0.7474 | OK |
| Transition energetique | 0.8023 | 0.6938 | BNP domine (langue) |

---

## Structure du projet

```
financial-rag-analyzer/
├── data/
│   ├── reports/          <- PDFs (non versionnes)
│   └── processed/        <- chunks.json + embeddings.npy
├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_chunking_embeddings.ipynb
│   ├── 03_vector_store.ipynb
│   └── 04_rag_pipeline.ipynb
├── src/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   └── generation/
├── app/
│   └── streamlit_app.py
├── reports/figures/
├── requirements.txt
└── README.md
```

---

## Lancer le projet

```bash
git clone https://github.com/TinhinaneBA/financial-rag-analyzer.git
cd financial-rag-analyzer

python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Installer Ollama : https://ollama.com
ollama pull mistral

# Telecharger les PDFs et les placer dans data/reports/
# Lancer les notebooks dans l ordre (01 → 04)

streamlit run app/streamlit_app.py
```

---

## Stack technique

| Categorie | Outil |
|---|---|
| Extraction PDF | pdfplumber |
| Chunking | LangChain TextSplitter |
| Embedding | Sentence-Transformers (all-MiniLM-L6-v2) |
| Retrieval | Cosine similarity numpy |
| LLM | Ollama Mistral 7B (local, gratuit) |
| Interface | Streamlit |
| Tracking | MLflow (notebooks) |

---

## Limitations et ameliorations futures

- **Multilingue :** TotalEnergies (EN) vs BNP (FR) — poser les questions
  dans la langue du rapport pour de meilleurs resultats
- **Performance LLM :** Mistral local lent sur CPU (700s+)
  — en production : GPT-3.5-turbo (3-5s)
- **Ajout de documents :** architecture modulaire — tout nouveau PDF
  peut etre integre en relancant les notebooks 01-02

---

## Auteure

**Tinhinane B.** — Etudiante M2 IWOCS, Universite Le Havre Normandie
[GitHub TinhinaneBA](https://github.com/TinhinaneBA)

---

*Projet 3 — Portfolio Data Science professionnel — 2025*