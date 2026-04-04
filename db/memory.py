import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.messages import HumanMessage

from utils.resilience import invoke_with_retry
from utils.prompt_loader import load_prompt

class SemanticMemory:
    """
    Manager for Long-Term Semantic Memory backed by a local vector database (Qdrant).
    Retains conversational history, condenses interactions, and retrieves relevant
    context based on vector similarity to maintain seamless chat fluidity.
    """
    def __init__(self, llm, embeddings):
        """
        Initializes the memory instance creating a temporary or persistent Qdrant client.

        Args:
            llm: Large Language Model used to compress lengthy interactions.
            embeddings: Vectorization model for storing and searching semantic text.
        """
        self.llm = llm
        self.embeddings = embeddings
        
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = "chat_history"
        
        dummy_vector = self.embeddings.embed_query("testing")
        self.vector_size = len(dummy_vector)
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        self.msg_count = 0

    async def add_interaction(self, session_id: str, user_msg: str, ai_msg: str, metadata: Dict[str, Any] = None):
        """
        Saves a conversation turn associating it with a specific session to isolate concurrent requests.
        """
        summary = await self._summarize_interaction(user_msg, ai_msg)
        vector  = self.embeddings.embed_query(summary)
        
        self.msg_count += 1
        payload = {
            "user": user_msg,
            "ai": ai_msg,
            "summary": summary,
            "timestamp": self.msg_count,
            "session_id": session_id
        }
        if metadata:
            payload.update(metadata)
            
        point_id = str(uuid.uuid4())
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )]
        )
        print(f"   🧠 Interaction saved to semantic memory (Session: {session_id[:8]})")

    async def _summarize_interaction(self, user: str, ai: str) -> str:
        raw_prompt = load_prompt("memory_summarizer.txt")
        prompt     = raw_prompt.format(USER=user, AI=ai)
        
        try:
            resp = await invoke_with_retry(self.llm, [HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception:
            return user[:50] + "..."

    def get_context(self, session_id: str, current_query: str, top_k: int = 3) -> str:
        """
        Scans the collection strictly filtering by the session's UUID.
        """
        if self.msg_count == 0:
            return ""
            
        from qdrant_client import models
        vector = self.embeddings.embed_query(current_query)
        _filter = models.Filter(must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))])

        if hasattr(self.qdrant, "query_points"):
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                query_filter=_filter
            ).points
        else:
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                query_filter=_filter
            )
        
        if not results:
            return ""
            
        sorted_results = sorted(results, key=lambda x: x.payload["timestamp"])
        
        context_str = "CONTEXTO HISTÓRICO DE ESTA CONVERSACIÓN:\n"
        for r in sorted_results:
            if r.score > 0.40:
                context_str += f"- Usuario: {r.payload['user']}\n"
                context_str += f"  IA: {r.payload['ai'][:150]}...\n"
                
        return context_str
