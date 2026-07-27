"""
vectorizer.py — Chunking et generation d'embeddings
Auteure : TinhinaneBA
"""

import json
import numpy as np
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Parametres par defaut
DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MODEL         = 'sentence-transformers/all-MiniLM-L6-v2'


def chunk_documents(all_docs: dict,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Decoupe les documents en chunks avec RecursiveCharacterTextSplitter.

    Args:
        all_docs     : dict {nom: contenu} issu de pdf_loader
        chunk_size   : taille max d'un chunk en caracteres
        chunk_overlap: chevauchement entre chunks

    Returns:
        liste de chunks avec metadonnees
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc_name, doc in all_docs.items():
        doc_chunks = []

        for page in doc['pages']:
            text = page.get('text_clean', page['text'])
            if len(text.strip()) < 50:
                continue

            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                doc_chunks.append({
                    'doc_name' : doc_name,
                    'filename' : doc['filename'],
                    'page'     : page['page'],
                    'chunk_id' : f"{doc_name}_p{page['page']}_c{i}",
                    'text'     : chunk,
                    'n_chars'  : len(chunk),
                    'n_words'  : len(chunk.split())
                })

        all_chunks.extend(doc_chunks)
        print(f"✅ {doc_name} — {len(doc_chunks):,} chunks crees")

    print(f"\nTotal : {len(all_chunks):,} chunks")
    return all_chunks


def generate_embeddings(chunks: list,
                        model_name: str = DEFAULT_MODEL,
                        batch_size: int = 64) -> np.ndarray:
    """
    Genere les embeddings pour une liste de chunks.

    Args:
        chunks     : liste de chunks
        model_name : nom du modele Sentence-Transformers
        batch_size : taille des batchs d'encodage

    Returns:
        numpy array de shape (n_chunks, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    texts = [c['text'] for c in chunks]

    print(f"Generation des embeddings...")
    print(f"   Modele    : {model_name}")
    print(f"   Dimension : {model.get_sentence_embedding_dimension()}")
    print(f"   Chunks    : {len(texts):,}")

    embeddings = model.encode(
        texts,
        batch_size        = batch_size,
        show_progress_bar = True,
        convert_to_numpy  = True
    )

    print(f"✅ Embeddings generes — shape : {embeddings.shape}")
    return embeddings


def save_chunks_and_embeddings(chunks: list,
                               embeddings: np.ndarray,
                               output_dir: Path) -> None:
    """
    Sauvegarde les chunks (JSON) et embeddings (numpy).

    Args:
        chunks     : liste de chunks
        embeddings : array numpy
        output_dir : dossier de sortie
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path    = output_dir / 'chunks.json'
    embeddings_path = output_dir / 'embeddings.npy'

    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(str(embeddings_path), embeddings)

    print(f"✅ Chunks sauvegardés      → {chunks_path}")
    print(f"✅ Embeddings sauvegardes  → {embeddings_path}")
    print(f"   Taille embeddings       : {embeddings.nbytes / 1e6:.1f} MB")


def load_chunks_and_embeddings(processed_dir: Path,
                               min_chars: int = 20):
    """
    Charge et filtre les chunks et embeddings sauvegardes.

    Args:
        processed_dir : dossier contenant chunks.json et embeddings.npy
        min_chars     : taille minimale d'un chunk valide

    Returns:
        tuple (chunks_filtres, embeddings_filtres)
    """
    with open(processed_dir / 'chunks.json', encoding='utf-8') as f:
        all_chunks = json.load(f)

    all_embeddings = np.load(str(processed_dir / 'embeddings.npy'))

    valid_idx  = [i for i, c in enumerate(all_chunks)
                  if c['n_chars'] >= min_chars]
    chunks     = [all_chunks[i] for i in valid_idx]
    embeddings = all_embeddings[valid_idx]

    print(f" {len(chunks):,} chunks charges (filtres depuis {len(all_chunks):,})")
    print(f"   Embeddings shape : {embeddings.shape}")

    return chunks, embeddings