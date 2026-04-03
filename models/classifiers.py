import json
import re
import numpy as np
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

from utils.config import ANCLAS_SEMANTICAS_HISTOLOGIA
from utils.resilience import invoke_con_reintento, embed_query_con_reintento, embed_documents_con_reintento

class ClasificadorSemantico:
    """
    Determina si una consulta pertenece al dominio histológico combinando:
    1. Similitud coseno entre un embedding de la consulta y texto de anclas semánticas predefinidas.
    2. Razonamiento estructurado por un LLM interactuando con el contexto visual disponible en el turno.
    Actúa como cortafuegos o "guardrail" evitando indexación innecesaria.
    """
    UMBRAL_SIMILITUD = 0.49
    UMBRAL_LLM       = 0.49

    def __init__(self, llm, embeddings, device: str, temario: List[str]):
        """
        Inicializa el filtro Semántico del dominio.

        Args:
            llm: Modelo conversacional analítico (LangChain).
            embeddings: Pipeline o modelo que provee la función embed_documents / embed_query.
            device (str): Tipo de hardware detectado ('cuda' o 'cpu').
            temario (List[str]): Lista general de temas conocidos aplicables del manual provisto.
        """
        self.llm       = llm
        self.embeddings = embeddings
        self.device    = device
        self.temario   = temario
        self._anclas_emb: Optional[np.ndarray] = None

    def _embed_textos(self, textos: List[str]) -> np.ndarray:
        """
        Convierte una lista de strings a arreglos numéricos persistentes por el modelo Embedding.

        Args:
            textos (List[str]): Matriz textual genérica.

        Returns:
            np.ndarray: Vectores resultantes (N dimensions).
        """
        return np.array(embed_documents_con_reintento(self.embeddings, textos))

    def _get_anclas_emb(self) -> np.ndarray:
        """
        Implementa un sigilo (Lazy initialization) para codificar a vectores un arreglo 
        constante de preguntas e intenciones representativas de la histología.

        Returns:
            np.ndarray: Vectorización de las anclas semánticas.
        """
        if self._anclas_emb is None:
            print("   🔄 Precalculando embeddings de anclas semánticas...")
            self._anclas_emb = self._embed_textos(ANCLAS_SEMANTICAS_HISTOLOGIA)
        return self._anclas_emb

    def similitud_con_dominio(self, consulta: str) -> float:
        """
        Rastrea algebraicamente si el embedding de una pregunta dada cruza los umbrales de 
        proximidad multidimensional junto con el corpus conocido de interés. 

        Args:
            consulta (str): Pregunta actual en turno.

        Returns:
            float: Puntaje indicando severidad simétrica de su relación (0 a 1).
        """
        try:
            q_emb = np.array(embed_query_con_reintento(self.embeddings, consulta))
            a_emb = self._get_anclas_emb()
            sims  = (q_emb @ a_emb.T).flatten()
            return float(np.max(sims))
        except Exception as e:
            print(f"   ⚠️ Error similitud semántica: {e}")
            return 0.0

    async def clasificar(
        self,
        consulta: str,
        analisis_visual: Optional[str] = None,
        imagen_activa: bool = False,
        temario_muestra: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Enruta o descarta intenciones. Es la puerta de entrada oficial del diagrama 
        de estados para evitar búsquedas RAG irrelevantes. 

        Args:
            consulta (str): Pregunta principal a evaluar.
            analisis_visual (Optional[str]): Análisis previo generado de una imagen cargada.
            imagen_activa (bool): Indicador binario dictando si estamos procesando visualmente un contexto.
            temario_muestra (Optional[List[str]]): Selección de las categorías relevantes para guiar el LLM si es necesario.

        Returns:
            Dict[str, Any]: Objeto validando el estado `valido`, `metodo` utilizado (semantico o llm) y `motivo` del descarte o aceptación.
        """
        sim = self.similitud_con_dominio(consulta)
        print(f"   📐 Similitud semántica con dominio histológico: {sim:.4f}")

        umbral_efectivo = self.UMBRAL_SIMILITUD * (0.6 if imagen_activa else 1.0)

        if sim >= umbral_efectivo:
            return {
                "valido":            True,
                "tema_encontrado":   None,
                "motivo":            f"Similitud semántica {sim:.3f} ≥ umbral {umbral_efectivo:.3f}",
                "similitud_dominio": sim,
                "metodo":            "semantico"
            }

        muestra_temas = (temario_muestra or self.temario)[:60]
        temario_txt   = "\n".join(f"- {t}" for t in muestra_temas)

        context_extra = ""
        if analisis_visual:
            context_extra = f"\n\nANÁLISIS DE IMAGEN DISPONIBLE:\n{analisis_visual[:600]}"
        if imagen_activa:
            context_extra += "\n\n[El usuario tiene una imagen histológica activa en el chat]"

        system = f"""Eres un clasificador de intención para un sistema RAG de histología médica.

Tu tarea: determinar si la consulta es una pregunta relacionada con histología,
patología, anatomía microscópica o morfología celular/tisular.

IMPORTANTE:
- "¿de qué tipo de tejido se trata?" SÍ es histológica.
- "¿qué ves en la imagen?" en contexto histológico SÍ es histológica.
- No es necesario que mencione palabras técnicas si el contexto lo indica.
- Si hay imagen histológica activa, dar beneficio de la duda.

TEMARIO DISPONIBLE (muestra):
{temario_txt}
{context_extra}

Responde ÚNICAMENTE en JSON válido (sin backticks):
{{"valido": true/false, "tema_encontrado": "tema más cercano o null", "confianza": 0.0-1.0, "motivo": "explicación breve"}}"""

        try:
            resp      = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=f"CONSULTA: {consulta}")
            ])
            texto     = re.sub(r"```json\s*|\s*```", "", resp.content.strip())
            data      = json.loads(texto)
            confianza = float(data.get("confianza", 0.49))
            valido    = bool(data.get("valido", True))

            if not valido and imagen_activa and confianza < 0.7:
                valido = True
                data["motivo"] += " [aceptado por imagen activa]"

            return {
                "valido":            valido,
                "tema_encontrado":   data.get("tema_encontrado"),
                "motivo":            data.get("motivo", ""),
                "similitud_dominio": sim,
                "metodo":            "llm" if sim < umbral_efectivo * 0.49 else "combinado"
            }
        except Exception as e:
            print(f"   ⚠️ Error clasificador LLM: {e}")
            return {
                "valido":            imagen_activa or sim > 0.10,
                "tema_encontrado":   None,
                "motivo":            f"Fallback: {e}",
                "similitud_dominio": sim,
                "metodo":            "fallback"
            }
