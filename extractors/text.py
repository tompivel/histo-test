import json
import re
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from utils.resilience import invoke_con_reintento

class ExtractorTemario:
    """
    Clase responsable de consultar al LLM para extraer y listar el temario 
    o índice de temas presentes en un corpus de texto histológico.
    """
    def __init__(self, llm):
        """
        Inicializa el Extractor de Temario.

        Args:
            llm: Modelo conversacional (ej. Groq / Llama) a utilizar.
        """
        self.llm   = llm
        self.temas: List[str] = []

    async def extraer_temario(self, texto_completo: str) -> List[str]:
        """
        Llama al API del LLM para consolidar un temario exhaustivo basado 
        en las primeras partes del texto. El resultado se cachea en un archivo JSON local.

        Args:
            texto_completo (str): Texto crudo volcado desde el manual Base.

        Returns:
            List[str]: Lista de temas identificados.
        """
        print("📋 Extrayendo temario...")
        muestra = texto_completo[:8000]
        system = (
            "Eres un experto en histología. Genera una lista EXHAUSTIVA de temas, "
            "estructuras, tejidos, células, tinciones del manual.\n"
            "Un tema por línea, sin bullets. Solo la lista."
        )
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=f"TEXTO:\n{muestra}")
            ])
            temas_raw  = resp.content.strip().split("\n")
            self.temas = [t.strip() for t in temas_raw if t.strip() and len(t.strip()) > 2]
            print(f"✅ Temario: {len(self.temas)} temas")
            with open("temario_histologia.json", "w", encoding="utf-8") as f:
                json.dump(self.temas, f, ensure_ascii=False, indent=2)
            return self.temas
        except Exception as e:
            print(f"❌ Error: {e}")
            return []

    def get_temario_texto(self) -> str:
        """
        Devuelve el formato texto stringificado del temario, con bullets Markdown.

        Returns:
            str: Temario textual formateado para inyectarse en los prompts.
        """
        return "\n".join(f"- {t}" for t in self.temas[:100]) if self.temas else "No disponible."


class ExtractorEntidades:
    """
    Clase dedicada al Procesamiento de Lenguaje Natural para extraer menciones a 
    tejidos, estructuras celulares y tinciones específicas dentro de strings de texto.
    """
    def __init__(self, llm):
        """
        Inicializa ExtractorEntidades.

        Args:
            llm: Agente LLM para extracción robusta de entidades.
        """
        self.llm = llm

    async def extraer_de_texto(self, texto: str) -> Dict[str, List[str]]:
        """
        Usa el LLM explícitamente para extraer con certeza entidades en consultas del usuario.

        Args:
            texto (str): Consulta natural del usuario.

        Returns:
            Dict[str, List[str]]: JSON parseado enumerando entidades clasificadas en:
                                  `tejidos`, `estructuras`, `tinciones`.
        """
        system = (
            "Extrae entidades histológicas del texto. "
            'Responde SOLO en JSON: {"tejidos": [...], "estructuras": [...], "tinciones": [...]}\n'
            "Máximo 3 items por categoría. Si no hay, lista vacía."
        )
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=texto[:500])
            ])
            texto_resp = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            resultado  = json.loads(texto_resp)
            return {
                "tejidos":     [t.lower() for t in resultado.get("tejidos", [])[:3]],
                "estructuras": [e.lower() for e in resultado.get("estructuras", [])[:3]],
                "tinciones":   [t.lower() for t in resultado.get("tinciones", [])[:3]],
            }
        except Exception:
            return {"tejidos": [], "estructuras": [], "tinciones": []}

    def extraer_de_texto_sync(self, texto: str) -> Dict[str, List[str]]:
        """
        Extrae entidades con operaciones síncronas usando búsqueda local hardcodeada.
        Ideal para procesar y categorizar veloces chunks de texto de los PDFs de manera masiva.

        Args:
            texto (str): Texto analizado del chunk.

        Returns:
            Dict[str, List[str]]: Categorización síncrona extraída sin latencia de red LLM.
        """
        entidades: Dict[str, List[str]] = {"tejidos": [], "estructuras": [], "tinciones": []}
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
        texto_lower = texto.lower()
        entidades["tejidos"]     = [t for t in TEJIDOS     if t in texto_lower][:3]
        entidades["estructuras"] = [e for e in ESTRUCTURAS if e in texto_lower][:3]
        entidades["tinciones"]   = [t for t in TINCIONES   if t in texto_lower][:3]
        return entidades
