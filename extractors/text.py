import json
import re
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from utils.resilience import invoke_with_retry
from utils.prompt_loader import load_prompt

class TopicExtractor:
    """
    Extracts and lists the topics/syllabus present within a raw histological corpus.
    """
    def __init__(self, llm):
        """
        Initializes the Topic Extractor.
        
        Args:
            llm: Conversational model used for logic extraction.
        """
        self.llm     = llm
        self.topics: List[str] = []

    async def extract_topics(self, full_text: str) -> List[str]:
        """
        Calls the LLM API to consolidate an exhaustive syllabus based on text.
        """
        print("📋 Extracting syllabus...")
        sample = full_text[:8000]
        system = load_prompt("topic_extractor.txt")
        try:
            resp = await invoke_with_retry(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=f"TEXTO:\n{sample}")
            ])
            topics_raw  = resp.content.strip().split("\n")
            self.topics = [t.strip() for t in topics_raw if t.strip() and len(t.strip()) > 2]
            print(f"✅ Syllabus {len(self.topics)} topics extracted")
            with open("temario_histologia.json", "w", encoding="utf-8") as f:
                json.dump(self.topics, f, ensure_ascii=False, indent=2)
            return self.topics
        except Exception as e:
            print(f"❌ Error: {e}")
            return []

    def get_topics_text(self) -> str:
        """
        Returns a stringified format of the syllabus with Markdown bullets.
        """
        return "\n".join(f"- {t}" for t in self.topics[:100]) if self.topics else "No disponible."


class EntityExtractor:
    """
    NLP module to extract tissues, anatomical structures, and stains from natural text.
    """
    def __init__(self, llm):
        self.llm = llm

    async def extract_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Explicitly leverages the LLM to classify entities in user queries.
        """
        system = load_prompt("entity_extractor.txt")
        try:
            resp = await invoke_with_retry(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=text[:700])
            ])
            clean_resp = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            result     = json.loads(clean_resp)
            return {
                "tejidos":     [t.lower() for t in result.get("tejidos", [])[:3]],
                "estructuras": [e.lower() for e in result.get("estructuras", [])[:3]],
                "tinciones":   [t.lower() for t in result.get("tinciones", [])[:3]],
            }
        except Exception:
            return {"tejidos": [], "estructuras": [], "tinciones": []}

    def extract_from_text_sync(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts entities using synchronous hardcoded keyword matching for rapid ingestion.
        """
        entities: Dict[str, List[str]] = {"tejidos": [], "estructuras": [], "tinciones": []}
        TEJIDOS = [
            "epitelio", "conectivo", "muscular", "nervioso", "cartílago", "hueso",
            "sangre", "linfoide", "hepático", "renal", "pulmonar", "dérmico",
            "epitelial", "estroma", "mucosa", "serosa"
        ]
        ESTRUCTURAS = [
            "célula", "núcleo", "citoplasma", "membrana", "gránulo", "fibra",
            "canalículo", "vellosidad", "cripta", "glomérulo", "túbulo", "alvéolo",
            "folículo", "sinusoide", "perla córnea", "cuerpo de albicans",
            "cuerpo de councilman", "queratina", "colágeno"
        ]
        TINCIONES = [
            "h&e", "hematoxilina", "eosina", "pas", "tricrómico", "grocott",
            "ziehl", "giemsa", "reticulina", "alcian blue", "von kossa"
        ]
        text_lower = text.lower()
        entities["tejidos"]     = [t for t in TEJIDOS     if t in text_lower][:3]
        entities["estructuras"] = [e for e in ESTRUCTURAS if e in text_lower][:3]
        entities["tinciones"]   = [t for t in TINCIONES   if t in text_lower][:3]
        return entities
