"""
Research Chat Session Manager.
Maintains multi-turn conversation state, chat history, and context-augmented queries.
"""

from typing import List, Dict, Any, Optional
from src.rag.rag_engine import RAGEngine

class ResearchChatSession:
    """Manages chat history and coordinates RAG retrieval across conversation turns."""

    def __init__(self, faiss_index: Any, embedder: Any):
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.history: List[Dict[str, str]] = []

    def send_message(self, user_message: str, top_k: int = 4) -> Dict[str, Any]:
        """Process user message, retrieve context from FAISS, generate answer, and update history."""
        # Append user message to history
        self.history.append({"role": "user", "content": user_message})

        # Execute RAG query
        rag_response = RAGEngine.query(
            user_query=user_message,
            faiss_index=self.faiss_index,
            embedder=self.embedder,
            top_k=top_k
        )

        assistant_answer = rag_response["answer"]
        self.history.append({"role": "assistant", "content": assistant_answer})

        return {
            "answer": assistant_answer,
            "citations": rag_response["citations"],
            "retrieved_chunks_count": rag_response["retrieved_chunks_count"],
            "history": self.history
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()

