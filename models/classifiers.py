import json
import re
import numpy as np
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import SEMANTIC_ANCHORS
from utils.resilience import invoke_with_retry, embed_query_with_retry, embed_documents_with_retry
from utils.prompt_loader import load_prompt

class SemanticClassifier:
    """
    Determines if a user query belongs to the histological domain by combining:
    1. Cosine similarity between the query embedding and predefined semantic anchors.
    2. Structured reasoning by an LLM interacting with visual context if available.
    Acts as a firewall or guardrail avoiding irrelevant RAG searches.
    """
    SIMILARITY_THRESHOLD = 0.49
    LLM_THRESHOLD        = 0.49

    def __init__(self, llm, embeddings, device: str, syllabus: List[str]):
        """
        Initializes the Semantic Domain Filter.

        Args:
            llm: Conversational analytics model (LangChain wrapper).
            embeddings: Pipeline providing embed_documents / embed_query.
            device str: Hardware device ('cuda' or 'cpu').
            syllabus List[str]: Known broad topics applicable from the manual.
        """
        self.llm        = llm
        self.embeddings = embeddings
        self.device     = device
        self.syllabus   = syllabus
        self._anchors_emb: Optional[np.ndarray] = None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Converts string arrays into numeric vectors via the Embedding pipeline.
        """
        return np.array(embed_documents_with_retry(self.embeddings, texts))

    def _get_anchors_emb(self) -> np.ndarray:
        """
        Lazy initialization for anchor embeddings mapping common histology intents.

        Returns:
            np.ndarray: Vectorized semantic boundaries.
        """
        if self._anchors_emb is None:
            print("   🔄 Pre-calculating Semantic Anchor embeddings...")
            self._anchors_emb = self._embed_texts(SEMANTIC_ANCHORS)
        return self._anchors_emb

    def domain_similarity(self, query: str) -> float:
        """
        Mathematically maps the query proximity to the nearest semantic anchor.

        Args:
            query (str): The current user prompt.

        Returns:
            float: Symmetrical proximity score (0.0 to 1.0).
        """
        try:
            q_emb = np.array(embed_query_with_retry(self.embeddings, query))
            a_emb = self._get_anchors_emb()
            sims  = (q_emb @ a_emb.T).flatten()
            return float(np.max(sims))
        except Exception as e:
            print(f"   ⚠️ Semantic similarity error: {e}")
            return 0.0

    async def classify(
        self,
        query: str,
        visual_analysis: Optional[str] = None,
        active_image: bool = False,
        syllabus_sample: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Routes or discards intents. This acts as the official entry gate to the state graph.

        Args:
            query (str): The primary prompt evaluated.
            visual_analysis (Optional[str]): Prior image extraction if uploaded.
            active_image (bool): True if visually constrained context exists.
            syllabus_sample (Optional[List[str]]): Sample of DB topics for LLM context.

        Returns:
            Dict[str, Any]: Validation payload dictating RAG proceedability.
        """
        sim = self.domain_similarity(query)
        print(f"   📐 Semantic Similarity to domain: {sim:.4f}")

        effective_threshold = self.SIMILARITY_THRESHOLD * (0.6 if active_image else 1.0)

        if sim >= effective_threshold:
            return {
                "valido":            True,
                "tema_encontrado":   None,
                "motivo":            f"Semantic Threshold {sim:.3f} >= {effective_threshold:.3f}",
                "similitud_dominio": sim,
                "metodo":            "semantic"
            }

        sample_topics = (syllabus_sample or self.syllabus)[:60]
        syllabus_txt  = "\n".join(f"- {t}" for t in sample_topics)

        context_extra = ""
        if visual_analysis:
            context_extra = f"\n\nANÁLISIS DE IMAGEN DISPONIBLE:\n{visual_analysis[:600]}"
        if active_image:
            context_extra += "\n\n[El usuario tiene una imagen histológica activa en el chat]"

        # Decoupled prompt injection
        raw_prompt = load_prompt("domain_classifier.txt")
        system     = raw_prompt.format(
            SYLLABUS_TXT=syllabus_txt,
            CONTEXT_EXTRA=context_extra
        )

        try:
            resp      = await invoke_with_retry(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=f"CONSULTA: {query}")
            ])
            text      = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            data      = json.loads(text)
            
            confidence = float(data.get("confianza", 0.49))
            is_valid   = bool(data.get("valido", True))

            if not is_valid and active_image and confidence < 0.7:
                is_valid = True
                data["motivo"] += " [Accepted via visual fallback active exception]"

            return {
                "valido":            is_valid,
                "tema_encontrado":   data.get("tema_encontrado"),
                "motivo":            data.get("motivo", ""),
                "similitud_dominio": sim,
                "metodo":            "llm" if sim < effective_threshold * 0.49 else "hybrid"
            }
        except Exception as e:
            print(f"   ⚠️ LLM Classifier fallback error: {e}")
            return {
                "valido":            active_image or sim > 0.10,
                "tema_encontrado":   None,
                "motivo":            f"Fallback Exception: {e}",
                "similitud_dominio": sim,
                "metodo":            "fallback"
            }
