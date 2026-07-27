"""
rag_chain.py — Pipeline RAG complet : retrieve → augment → generate
Auteure : TinhinaneBA
"""

import time
from langchain_ollama import OllamaLLM

SYSTEM_PROMPT = """Tu es un analyste financier expert specialise dans
l analyse de rapports annuels d entreprises.

REGLES ABSOLUES :
1. Tu reponds UNIQUEMENT avec les informations du contexte fourni
2. Tu NE dois PAS utiliser tes connaissances generales
3. Si l information n est PAS dans le contexte, reponds :
   Cette information n est pas disponible dans les documents fournis.
4. Tu cites TOUJOURS tes sources avec [Source N — page X]
5. Tu n inventes AUCUNE URL ni reference externe
6. Tu reponds dans la meme langue que la question"""


class RAGChain:
    """
    Pipeline RAG complet.
    Orchestre le retrieval et la generation de reponses.
    """

    def __init__(self,
                 retriever,
                 model_name: str = "mistral",
                 temperature: float = 0.1):
        """
        Args:
            retriever   : instance de SemanticRetriever
            model_name  : modele Ollama a utiliser
            temperature : temperature du LLM (0.1 = factuel)
        """
        self.retriever = retriever
        self.llm       = OllamaLLM(model=model_name,
                                   temperature=temperature)
        self.model_name = model_name
        print(f"✅ RAGChain initialise — LLM : {model_name}")

    def _build_context(self, chunks: list) -> str:
        """Construit le contexte a partir des chunks recuperes."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_short = chunk['doc_name'][:30]
            parts.append(
                f"[Source {i} — {doc_short}, page {chunk['page']}]\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def answer(self,
               query: str,
               n_chunks: int = 5,
               doc_filter: str = None) -> dict:
        """
        Pipeline RAG complet : retrieve → augment → generate.

        Args:
            query      : question en langage naturel
            n_chunks   : nombre de chunks a recuperer
            doc_filter : filtrer par document (nom exact)

        Returns:
            dict avec reponse, sources, modele et temps
        """
        t0 = time.time()

        # 1. RETRIEVE
        chunks  = self.retriever.retrieve(
            query, n_results=n_chunks, doc_filter=doc_filter
        )

        # 2. AUGMENT
        context = self._build_context(chunks)
        prompt  = f"""{SYSTEM_PROMPT}

CONTEXTE EXTRAIT DES RAPPORTS FINANCIERS :
{context}

QUESTION : {query}

REPONSE :"""

        # 3. GENERATE
        answer  = self.llm.invoke(prompt)
        elapsed = round(time.time() - t0, 1)

        return {
            'query'  : query,
            'answer' : answer,
            'sources': chunks,
            'model'  : self.model_name,
            'time_s' : elapsed,
            'n_chunks': len(chunks)
        }

    def batch_answer(self, queries: list, **kwargs) -> list:
        """
        Repond a une liste de questions.

        Args:
            queries : liste de questions
            **kwargs: arguments passes a answer()

        Returns:
            liste de resultats
        """
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Question {i}/{len(queries)} : {query[:50]}...")
            result = self.answer(query, **kwargs)
            results.append(result)
            print(f"  -> {result['time_s']}s")
        return results