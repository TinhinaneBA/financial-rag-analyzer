"""
Financial RAG Analyzer — Streamlit Cloud Version
Auteure : TinhinaneBA
LLM : HuggingFace Inference API (Mistral-7B-Instruct)
"""

import streamlit as st
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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
# HuggingFace Inference API
# ══════════════════════════════════════════════════════════════

def generate_hf(prompt: str) -> str:
    """Genere une reponse via HuggingFace Inference API."""
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        return "Erreur : token HuggingFace manquant dans les secrets Streamlit."

    API_URL = (
        "https://api-inference.huggingface.co/models/"
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs"    : prompt,
        "parameters": {
            "max_new_tokens"  : 600,
            "temperature"     : 0.1,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers,
                                 json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            return str(result)
        elif response.status_code == 503:
            return ("Le modele est en cours de chargement sur HuggingFace "
                    "(cold start). Veuillez reessayer dans 30 secondes.")
        else:
            return f"Erreur API HuggingFace : {response.status_code}"
    except requests.Timeout:
        return "Timeout — le modele met trop de temps a repondre. Reessayez."
    except Exception as e:
        return f"Erreur : {e}"


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES RESSOURCES
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_resources():
    base_dir   = Path(__file__).parent.parent
    processed  = base_dir / 'data' / 'processed'

    with open(processed / 'chunks.json', encoding='utf-8') as f:
        all_chunks = json.load(f)

    all_embeddings = np.load(str(processed / 'embeddings.npy'))

    valid_idx  = [i for i, c in enumerate(all_chunks)
                  if c['n_chars'] >= 20]
    chunks     = [all_chunks[i] for i in valid_idx]
    embeddings = all_embeddings[valid_idx]

    embed_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2'
    )

    return chunks, embeddings, embed_model


try:
    chunks, embeddings, embed_model = load_resources()
    RESOURCES_OK = True
except Exception as e:
    RESOURCES_OK = False
    st.error(f"Erreur de chargement : {e}")
    chunks, embeddings, embed_model = [], None, None


# ══════════════════════════════════════════════════════════════
# FONCTIONS RAG
# ══════════════════════════════════════════════════════════════

DOC_NAMES = {
    'bnp'  : 'bnp-paribas-ri-2023-fr-web',
    'total': ('totalenergies_universal-registration-'
              'document-2023_2023_en_pdf')
}


def retrieve_chunks(query, n_results=5, doc_filter=None):
    query_emb = embed_model.encode([query])
    scores    = cosine_similarity(query_emb, embeddings)[0]

    if doc_filter and doc_filter in DOC_NAMES:
        indices = [i for i, c in enumerate(chunks)
                   if c['doc_name'] == DOC_NAMES[doc_filter]]
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
        f"[Source {i+1} — {c['doc_name'][:25]}, page {c['page']}]\n"
        f"{c['text']}"
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
    answer = generate_hf(prompt)
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
    st.markdown("URD 2023 — 674 pages — EN")
    st.divider()

    st.markdown("### Parametres RAG")
    doc_choice = st.selectbox(
        "Filtrer par document",
        options=["Tous les documents", "BNP Paribas", "TotalEnergies"],
        index=0
    )
    n_chunks_param = st.slider(
        "Chunks recuperes (k)",
        min_value=1, max_value=8, value=5
    )
    st.divider()

    st.markdown("### Exemples de questions")
    examples = [
        "Quelle est la strategie de BNP Paribas pour 2025 ?",
        "What are TotalEnergies investments in renewables?",
        "Quels sont les indicateurs financiers de BNP ?",
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
- **Retrieval :** Cosine similarity
- **LLM :** Mistral-7B via HuggingFace API
- **Framework :** LangChain
    """)


# ══════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-title">📊 Financial RAG Analyzer</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Questions-reponses sur rapports financiers '
    '— RAG · Cosine Similarity · Mistral · HuggingFace</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "💬 Question & Reponse",
    "📈 Performance",
    "📖 A propos"
])


# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### Posez votre question sur les rapports financiers")

    st.markdown("""
    <div class="warning-box">
    Le rapport TotalEnergies est en anglais — posez vos questions
    en anglais pour de meilleurs resultats.
    Le rapport BNP Paribas est en francais.
    </div>
    """, unsafe_allow_html=True)

    default_q = st.session_state.get('question', '')
    question  = st.text_area(
        "Votre question",
        value=default_q,
        height=80,
        placeholder="Ex: Quelle est la strategie de BNP Paribas ?"
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
        with st.spinner("Analyse en cours via HuggingFace API..."):
            result = rag_answer(question,
                                n_chunks=n_chunks_param,
                                doc_filter=doc_filter)

        st.markdown("#### Reponse")
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f"Temps : **{result['time']}s** "
            f"| Chunks : **{len(result['sources'])}**"
        )

        st.markdown("#### Sources utilisees")
        for i, s in enumerate(result['sources'], 1):
            doc_label = "BNP Paribas" if "bnp" in s['doc_name'] \
                        else "TotalEnergies"
            with st.expander(
                f"Source {i} — {doc_label} · "
                f"page {s['page']} · score {s['score']}"
            ):
                st.markdown(s['text'])

    elif submit and not question.strip():
        st.warning("Veuillez entrer une question.")

    elif submit and not RESOURCES_OK:
        st.error("Les ressources ne sont pas chargees.")


# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Performance du systeme RAG")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks indexes",
              f"{len(chunks):,}" if RESOURCES_OK else "N/A")
    c2.metric("Documents",       "2 rapports")
    c3.metric("Dimensions emb.", "384")
    c4.metric("LLM",             "Mistral-7B HF")

    st.divider()
    st.markdown("#### Resultats des tests")

    df_tests = pd.DataFrame({
        'Test'        : ['BNP Strategie 2025',
                         'TotalE Renouvelables',
                         'Transition energetique'],
        'Score max'   : [0.7559, 0.7667, 0.8023],
        'Score moyen' : [0.6854, 0.7474, 0.6938],
        'Statut'      : ['OK', 'OK', 'BNP domine (langue)']
    })
    st.dataframe(df_tests, use_container_width=True, hide_index=True)

    st.divider()
    st.info("""
**Limitation multilingue :** TotalEnergies (EN) vs BNP (FR).
Posez les questions en anglais pour TotalEnergies.

**Cold start HuggingFace :** la premiere requete peut prendre
20-30 secondes le temps que le modele se charge.
    """)


# ── TAB 3 ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### A propos du projet")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### Objectif
Questions-reponses en langage naturel sur des rapports
financiers PDF avec citations exactes des sources.

#### Pipeline RAG
1. Extraction PDF (pdfplumber)
2. Chunking (500 chars, overlap 50)
3. Embedding (all-MiniLM-L6-v2, 384 dims)
4. Retrieval cosine similarity
5. Generation (Mistral-7B via HuggingFace)

#### Documents
- BNP Paribas — Rapport Integre 2023 (43 pages, FR)
- TotalEnergies — URD 2023 (674 pages, EN)
        """)

    with col2:
        st.markdown("""
#### Stack technique
| Composant | Outil |
|---|---|
| Extraction | pdfplumber |
| Chunking | LangChain |
| Embedding | Sentence-Transformers |
| Retrieval | Cosine similarity |
| LLM | Mistral-7B (HuggingFace) |
| Interface | Streamlit |

#### Auteure
**Tinhinane B.**
M2 IWOCS — Universite Le Havre Normandie

[GitHub](https://github.com/TinhinaneBA/financial-rag-analyzer)
        """)