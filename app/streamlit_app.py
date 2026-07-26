"""
Financial RAG Analyzer — Streamlit Application
Auteure : TinhinaneBA
Systeme de questions-reponses sur rapports financiers
RAG : Numpy + Sentence-Transformers + Ollama Mistral
"""

import streamlit as st
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

st.set_page_config(
    page_title="Financial RAG Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #f8f9fa;
        border-left: 4px solid #4f8ef7;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES RESSOURCES
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_resources():
    base_dir   = Path(__file__).parent.parent
    processed  = base_dir / 'data' / 'processed'

    # Charger chunks
    with open(processed / 'chunks.json', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Charger embeddings
    all_embeddings = np.load(str(processed / 'embeddings.npy'))

    # Filtrer chunks vides
    valid_idx  = [i for i, c in enumerate(all_chunks) if c['n_chars'] >= 20]
    chunks     = [all_chunks[i] for i in valid_idx]
    embeddings = all_embeddings[valid_idx]

    # Modele embedding
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # LLM
    llm = OllamaLLM(model="mistral", temperature=0.1)

    return chunks, embeddings, embed_model, llm


try:
    chunks, embeddings, embed_model, llm = load_resources()
    RESOURCES_OK = True
except Exception as e:
    RESOURCES_OK = False
    st.error(f"Erreur de chargement : {e}")
    chunks, embeddings, embed_model, llm = [], None, None, None


# ══════════════════════════════════════════════════════════════
# FONCTIONS RAG
# ══════════════════════════════════════════════════════════════

DOC_NAMES = {
    'bnp'  : 'bnp-paribas-ri-2023-fr-web',
    'total': 'totalenergies_universal-registration-document-2023_2023_en_pdf'
}


def retrieve_chunks(query, n_results=5, doc_filter=None):
    query_emb = embed_model.encode([query])
    scores    = cosine_similarity(query_emb, embeddings)[0]

    if doc_filter and doc_filter in DOC_NAMES:
        doc_name = DOC_NAMES[doc_filter]
        indices  = [i for i, c in enumerate(chunks)
                    if c['doc_name'] == doc_name]
    else:
        indices = list(range(len(chunks)))

    ranked = sorted(indices, key=lambda i: scores[i], reverse=True)
    top_k  = ranked[:n_results]

    return [{
        'text'    : chunks[i]['text'],
        'doc_name': chunks[i]['doc_name'],
        'page'    : chunks[i]['page'],
        'score'   : round(float(scores[i]), 4)
    } for i in top_k]


def rag_answer(query, n_chunks=5, doc_filter=None):
    retrieved = retrieve_chunks(query, n_results=n_chunks,
                                doc_filter=doc_filter)
    context   = "\n\n---\n\n".join([
        f"[Source {i+1} — {c['doc_name'][:25]}, page {c['page']}]\n{c['text']}"
        for i, c in enumerate(retrieved)
    ])
    prompt = f"""Tu es un analyste financier expert.

REGLES ABSOLUES :
1. Tu reponds UNIQUEMENT avec les informations du contexte ci-dessous
2. Tu NE dois PAS utiliser tes connaissances generales
3. Si l information n est PAS dans le contexte, reponds :
   Cette information n est pas disponible dans les documents fournis.
4. Tu cites TOUJOURS tes sources avec [Source N]
5. Tu n inventes AUCUNE URL ni reference externe

CONTEXTE :
{context}

QUESTION : {query}

REPONSE :"""

    t0     = time.time()
    answer = llm.invoke(prompt)
    elapsed = round(time.time() - t0, 1)

    return {'answer': answer, 'sources': retrieved, 'time': elapsed}


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Financial RAG Analyzer")
    st.markdown("**Auteure :** TinhinaneBA")
    st.markdown("**M2 IWOCS** — Universite Le Havre Normandie")
    st.divider()

    if RESOURCES_OK:
        st.success(f"✅ {len(chunks):,} chunks charges")
    else:
        st.error("Ressources non chargees")

    st.markdown("### Documents disponibles")
    st.markdown("**BNP Paribas**")
    st.markdown("Rapport Integre 2023 — 43 pages — FR")
    st.markdown("**TotalEnergies**")
    st.markdown("Universal Registration 2023 — 674 pages — EN")
    st.divider()

    st.markdown("### Parametres RAG")
    doc_choice = st.selectbox(
        "Filtrer par document",
        options=["Tous les documents", "BNP Paribas", "TotalEnergies"],
        index=0
    )
    n_chunks_param = st.slider(
        "Chunks recuperes (k)",
        min_value=1, max_value=10, value=5
    )
    st.divider()

    st.markdown("### Exemples de questions")
    examples = [
        "Quelle est la strategie de BNP Paribas pour 2025 ?",
        "What are TotalEnergies investments in renewables?",
        "Quels sont les indicateurs financiers cles de BNP ?",
        "How does TotalEnergies manage climate risks?",
        "Quelle est la politique RSE de BNP Paribas ?"
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state['question'] = ex

    st.divider()
    st.markdown("### Stack technique")
    st.markdown("""
- **Embedding :** all-MiniLM-L6-v2
- **Retrieval :** Cosine similarity (numpy)
- **LLM :** Ollama Mistral 7B
- **Framework :** LangChain
    """)


# ══════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">📊 Financial RAG Analyzer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Systeme de questions-reponses sur rapports '
    'financiers — RAG · Cosine Similarity · Mistral · LangChain</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "💬 Question & Reponse",
    "📈 Performance du systeme",
    "📖 A propos"
])


# ── TAB 1 — Q&A ──────────────────────────────────────────────
with tab1:
    st.markdown("### Posez votre question sur les rapports financiers")

    st.markdown("""
    <div class="warning-box">
    Le rapport TotalEnergies est en anglais — posez vos questions
    en anglais pour de meilleurs resultats sur ce document.
    Le rapport BNP Paribas est en francais.
    </div>
    """, unsafe_allow_html=True)

    default_q = st.session_state.get('question', '')
    question  = st.text_area(
        "Votre question",
        value=default_q,
        height=80,
        placeholder="Ex: Quelle est la strategie de BNP Paribas pour 2025 ?"
    )

    filter_map = {
        "Tous les documents": None,
        "BNP Paribas"       : "bnp",
        "TotalEnergies"     : "total"
    }
    doc_filter = filter_map[doc_choice]

    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("Analyser", type="primary",
                           use_container_width=True)

    if submit and question.strip() and RESOURCES_OK:
        with st.spinner("Recherche en cours... Mistral peut prendre "
                        "quelques minutes en local."):
            result = rag_answer(question,
                                n_chunks=n_chunks_param,
                                doc_filter=doc_filter)

        st.markdown("#### Reponse")
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"Temps : **{result['time']}s** "
                    f"| Chunks : **{len(result['sources'])}**")

        st.markdown("#### Sources utilisees")
        for i, s in enumerate(result['sources'], 1):
            doc_label = "BNP Paribas" if "bnp" in s['doc_name'] \
                        else "TotalEnergies"
            with st.expander(
                f"Source {i} — {doc_label} · page {s['page']} "
                f"· score {s['score']}"
            ):
                st.markdown(s['text'])

    elif submit and not question.strip():
        st.warning("Veuillez entrer une question.")

    elif submit and not RESOURCES_OK:
        st.error("Les ressources ne sont pas chargees correctement.")


# ── TAB 2 — PERFORMANCE ──────────────────────────────────────
with tab2:
    st.markdown("### Performance du systeme RAG")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks indexes",  f"{len(chunks):,}" if RESOURCES_OK else "N/A")
    col2.metric("Documents",       "2 rapports")
    col3.metric("Dimensions emb.", "384")
    col4.metric("Modele LLM",      "Mistral 7B")

    st.divider()
    st.markdown("#### Resultats des tests (notebooks)")

    df_tests = pd.DataFrame({
        'Test'        : ['BNP Strategie 2025',
                         'TotalE Renouvelables',
                         'Transition energetique'],
        'Score max'   : [0.7559, 0.7667, 0.8023],
        'Score moyen' : [0.6854, 0.7474, 0.6938],
        'Temps (s)'   : [742, 764, 2162],
        'Statut'      : ['OK', 'OK', 'BNP domine (langue)']
    })
    st.dataframe(df_tests, use_container_width=True, hide_index=True)

    st.divider()
    st.info("""
**Limitation multilingue :** TotalEnergies (anglais) vs BNP (francais).
Posez les questions en anglais pour TotalEnergies.

**Performance Mistral local :** 700s a 2000s sur CPU.
En production : utiliser GPT-3.5-turbo (3-5s).
    """)


# ── TAB 3 — A PROPOS ─────────────────────────────────────────
with tab3:
    st.markdown("### A propos du projet")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### Objectif
Systeme de questions-reponses en langage naturel sur des rapports
financiers PDF, avec citations des sources exactes.

#### Pipeline RAG
1. Extraction PDF (pdfplumber)
2. Chunking (RecursiveCharacterTextSplitter, 500 chars)
3. Embedding (all-MiniLM-L6-v2, 384 dims)
4. Indexation numpy (cosine similarity)
5. Retrieval (top-k chunks)
6. Generation (Ollama Mistral 7B)

#### Documents analyses
- BNP Paribas — Rapport Integre 2023 (43 pages, FR)
- TotalEnergies — URD 2023 (674 pages, EN)
        """)

    with col2:
        st.markdown("""
#### Stack technique
| Composant | Outil |
|---|---|
| Extraction PDF | pdfplumber |
| Chunking | LangChain TextSplitter |
| Embedding | Sentence-Transformers |
| Retrieval | Cosine similarity numpy |
| LLM | Ollama Mistral 7B |
| Interface | Streamlit |

#### Auteure
**Tinhinane B.**
M2 IWOCS — Universite Le Havre Normandie
Portfolio Data Science — 2025

[GitHub](https://github.com/TinhinaneBA/financial-rag-analyzer)
        """)