"""
Context-Aware RAG Engine.
Combines FAISS semantic search with Gemini generative models to provide evidence-grounded answers.
Combines FAISS semantic search with NVIDIA foundation models to provide evidence-grounded answers.
"""

import logging
from typing import List, Dict, Any, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    """Orchestrates retrieval from FAISS vector index and grounded answer generation."""

    @classmethod
    def query(
        cls,
        user_query: str,
        faiss_index: Any,
        embedder: Any,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Embed user query
        2. Retrieve top-K relevant chunks via FAISS
        3. Synthesize grounded answer with paper citations
        """
        if faiss_index.count() == 0:
            return {
                "answer": "The research corpus index is currently empty. Please run an analysis or upload papers first.",
                "citations": [],
                "retrieved_chunks_count": 0
            }

        query_emb = embedder.embed_text(user_query)
        retrieved_chunks = faiss_index.search(query_emb, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant paper sections were found matching your query.",
                "citations": [],
                "retrieved_chunks_count": 0
            }

        # Build context from chunks
        context_blocks = []
        citations = []
        seen_papers = set()

        for idx, chunk in enumerate(retrieved_chunks):
            p_id = chunk.get("paper_id")
            title = chunk.get("paper_title")
            year = chunk.get("year")
            sec = chunk.get("section_type", "excerpt")
            text = chunk.get("text", "")
            sim = chunk.get("similarity_score", 0.0)

            context_blocks.append(f"[{idx+1}] \"{title}\" ({year}) [{sec.upper()}]:\n{text}")

            if p_id not in seen_papers:
                citations.append({
                    "paper_id": p_id,
                    "title": title,
                    "year": year,
                    "section": sec,
                    "similarity": sim,
                    "doi": chunk.get("doi")
                })
                seen_papers.add(p_id)

        context_str = "\n\n".join(context_blocks)

        # Attempt LLM generation via NVIDIA API
        answer = None
        if settings.DYNAMIC_LLM_ENABLED and settings.NVIDIA_API_KEY:
            answer = cls._generate_with_llm(user_query, context_str)
        elif settings.LLM_REQUIRED:
            raise RuntimeError("NVIDIA_API_KEY is required because LLM_REQUIRED=true.")

        if not answer:
            answer = cls._generate_fallback(user_query, retrieved_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks_count": len(retrieved_chunks)
        }

    @classmethod
    def _generate_with_llm(cls, user_query: str, context_str: str) -> Optional[str]:
        """Query NVIDIA AI foundation models with grounded RAG prompt."""
        from src.llm.nvidia_client import NvidiaClient

        prompt = f"""You are ResearchGapAI's principal scientific research assistant.
Answer the user's research question strictly based on the following retrieved scientific context excerpts.
Cite relevant papers using their numbers (e.g. [1], [2]) directly in your sentences.
If the retrieved papers do not contain enough information, state what is known and specify what is missing.

RETRIEVED CONTEXT:
{context_str}

USER RESEARCH QUESTION:
{user_query}

Synthesize a clear, authoritative, and structured scientific answer:
"""
        try:
            return NvidiaClient.generate(prompt=prompt, temperature=settings.LLM_STRUCTURED_TEMPERATURE, max_tokens=2048)
        except Exception as e:
            logger.warning(f"RAG NVIDIA LLM generation failed: {e}. Falling back to structured synthesis.")
            return None

    @classmethod
    def _generate_fallback(cls, user_query: str, chunks: List[Dict[str, Any]]) -> str:
        """Deterministic structured synthesis when LLM is offline or quota reached."""
        top_chunks = chunks[:3]
        synthesis_lines = [
            f"Based on the retrieved academic corpus, here is the relevant evidence for '{user_query}':\n"
        ]
        for idx, c in enumerate(top_chunks):
            title = c.get("paper_title")
            year = c.get("year")
            sec = c.get("section_type", "section").title()
            excerpt = c.get("text", "")[:280]
            synthesis_lines.append(f"• **[{idx+1}] {title} ({year}) — {sec}:**\n  > \"{excerpt}...\"\n")

        synthesis_lines.append(
            "These findings highlight how current literature addresses this topic across the indexed publications."
        )
        return "\n".join(synthesis_lines)
