"""
retriever.py — Recherche semantique par similarite cosinus
Auteure : TinhinaneBA
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class SemanticRetriever:
    """
    Retriever semantique base sur la similarite cosinus.
    Charge les chunks et embeddings en memoire pour une
    recherche ultra-rapide sans dependance externe.
    """

    def __init__(self,
                 chunks: list,
                 embeddings: np.ndarray,
                 model_name: str = DEFAULT_MODEL):
        """
        Args:
            chunks     : liste de chunks avec metadonnees
            embeddings : array numpy (n_chunks, embedding_dim)
            model_name : modele Sentence-Transformers
        """
        self.chunks     = chunks
        self.embeddings = embeddings
        self.model      = SentenceTransformer(model_name)

        print(f"✅ SemanticRetriever initialise")
        print(f"   Chunks    : {len(chunks):,}")
        print(f"   Embedding : {embeddings.shape}")

    def retrieve(self,
                 query: str,
                 n_results: int = 5,
                 doc_filter: str = None) -> list:
        """
        Recherche les chunks les plus pertinents pour une question.

        Args:
            query      : question en langage naturel
            n_results  : nombre de chunks a retourner
            doc_filter : nom exact du document pour filtrer (optionnel)

        Returns:
            liste de chunks ordonnes par pertinence decroissante
        """
        # Encoder la question
        query_emb = self.model.encode([query])
        scores    = cosine_similarity(query_emb, self.embeddings)[0]

        # Filtrer par document si demande
        if doc_filter:
            indices = [i for i, c in enumerate(self.chunks)
                       if c['doc_name'] == doc_filter]
        else:
            indices = list(range(len(self.chunks)))

        # Trier par score decroissant
        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)
        top_k  = ranked[:n_results]

        return [{
            'text'    : self.chunks[i]['text'],
            'doc_name': self.chunks[i]['doc_name'],
            'filename': self.chunks[i]['filename'],
            'page'    : self.chunks[i]['page'],
            'score'   : round(float(scores[i]), 4)
        } for i in top_k]

    def get_available_documents(self) -> list:
        """Retourne la liste des documents uniques indexes."""
        return list(set(c['doc_name'] for c in self.chunks))

    def get_stats(self) -> dict:
        """Retourne les statistiques du retriever."""
        docs = self.get_available_documents()
        return {
            'total_chunks': len(self.chunks),
            'n_documents' : len(docs),
            'documents'   : docs,
            'embedding_dim': self.embeddings.shape[1]
        }


def build_retriever(processed_dir, model_name=DEFAULT_MODEL):
    """
    Factory function — construit un SemanticRetriever
    depuis les fichiers sauvegardes.

    Args:
        processed_dir : Path vers data/processed/
        model_name    : modele d'embedding

    Returns:
        instance de SemanticRetriever
    """
    from src.embeddings.vectorizer import load_chunks_and_embeddings
    from pathlib import Path

    chunks, embeddings = load_chunks_and_embeddings(Path(processed_dir))
    return SemanticRetriever(chunks, embeddings, model_name)