import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.messages import HumanMessage

from utils.resilience import invoke_con_reintento

class SemanticMemory:
    """
    Gestor de la Memoria Semántica a largo plazo apoyado por una base de datos vectorial local (Qdrant).
    Conserva el historial de la conversación, condensa las interacciones y recupera contexto 
    relevante basado en similitud vectorial para mantener fluidez en el chat.
    """
    def __init__(self, llm, embeddings):
        """
        Inicializa la instancia de memoria creando un cliente Qdrant temporal o persistente.

        Args:
            llm: Modelo grande de lenguaje usado para comprimir interacciones.
            embeddings: Modelo de vectorización para almacenar y buscar el texto.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.session_id = str(uuid.uuid4())
        
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = "chat_history"
        
        dummy_vector = self.embeddings.embed_query("testing")
        self.vector_size = len(dummy_vector)
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        self.msg_count = 0

    async def add_interaction(self, user_msg: str, ai_msg: str, metadata: Dict[str, Any] = None):
        """
        Guarda un turno de conversación (pregunta-respuesta) generando primero un resumen 
        ejecutivo en el LLM y luego vectorizándolo hacia la base de datos Qdrant.

        Args:
            user_msg (str): El prompt crudo ingresado por el usuario.
            ai_msg (str): La respuesta entregada por el asistente.
            metadata (Dict[str, Any], opcional): Metadatos adicionales para trazar el log.
        """
        resumen = await self._summarize_interaction(user_msg, ai_msg)
        vector = self.embeddings.embed_query(resumen)
        
        payload = {
            "user": user_msg,
            "ai": ai_msg,
            "summary": resumen,
            "timestamp": self.msg_count,
            "session_id": self.session_id
        }
        if metadata:
            payload.update(metadata)
            
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=self.msg_count,
                vector=vector,
                payload=payload
            )]
        )
        self.msg_count += 1
        print(f"   🧠 Interacción guardada en memoria semántica (ID: {self.msg_count-1})")

    async def _summarize_interaction(self, user: str, ai: str) -> str:
        """
        Refina y acorta largas interacciones médicas guardando sólo su esencia algorítmica.

        Args:
            user (str): La entrada del usuario original.
            ai (str): Respuesta extensa original RAG.

        Returns:
            str: Resumen del núcleo conversacional de no más de un par de oraciones.
        """
        prompt = f"""Resume de forma extremadamente concisa el siguiente intercambio. 
        Enfócate en la intención del usuario y la respuesta clave.
        Máximo 2 oraciones.
        
        User: {user}
        AI: {ai}
        
        Resumen:"""
        try:
            resp = await invoke_con_reintento(self.llm, [HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception:
            return user[:50] + "..."

    def get_context(self, current_query: str, top_k: int = 3) -> str:
        """
        Escanea la colección local Qdrant comparando la pregunta nueva contra 
        conversaciones consolidadas anteriores devolviendo los K historiales más parecidos.

        Args:
            current_query (str): Pregunta actual disparando la búsqueda.
            top_k (int, opcional): Límite de extracciones permitidas.

        Returns:
            str: Una cadena formateada inyectable que provee referencias sobre qué se estuvo hablando.
        """
        if self.msg_count == 0:
            return ""
            
        vector = self.embeddings.embed_query(current_query)
        if hasattr(self.qdrant, "query_points"):
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k
            ).points
        else:
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k
            )
        
        if not results:
            return ""
            
        sorted_results = sorted(results, key=lambda x: x.payload["timestamp"])
        
        context_str = "CONTEXTO HISTÓRICO DE ESTA MISMA CONVERSACIÓN:\n"
        for r in sorted_results:
            if r.score > 0.40:  # Umbral muy bajo para recuperar contexto general
                context_str += f"- User dijo: {r.payload['user']}\n"
                context_str += f"  AI respondió: {r.payload['ai'][:150]}...\n"
                
        return context_str
