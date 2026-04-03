import os
import io
from typing import TypedDict, Annotated, List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, END

# Importar utils y config
from utils.config import userdata, _safe, traceable
from utils.resilience import invoke_con_reintento
# Importar base de datos y memoria
from db.neo4j_client import Neo4jClient
from db.memory import SemanticMemory
# Importar clasificadores e imágenes
from models.classifiers import ClasificadorSemantico
from models.vision import PlipWrapper, UniWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

class AgentState(TypedDict):
    """
    Representa el estado global que fluye a través del grafo de LangGraph.
    
    Attributes:
        input (str): La pregunta o consulta original del usuario.
        chat_history (List[BaseMessage]): Historial de mensajes de la conversación actual.
        imagen_usuario (Optional[Image.Image]): Imagen proporcionada por el usuario en el turno actual (si hay).
        analisis_visual (Optional[str]): Análisis de la imagen extraído mediante un LLM de visión.
        contexto_db (List[Dict]): Resultados híbridos extraídos de la base de datos (Neo4j).
        memoria_relevante (str): Contexto histórico recuperado de Qdrant/Memoria Semántica.
        clasificacion_dominio (Dict[str, Any]): Resultado del clasificador semántico indicando si la pregunta es válida.
        output (str): Respuesta final generada por el agente para presentar al usuario.
    """
    input: str
    chat_history: List[BaseMessage]
    imagen_usuario: Optional[Image.Image]
    analisis_visual: Optional[str]
    contexto_db: List[Dict]
    memoria_relevante: str
    clasificacion_dominio: Dict[str, Any]
    output: str

class AsistenteHistologiaNeo4j:
    """
    Orquestador principal del sistema RAG usando LangGraph.
    
    Gestiona la inicialización de bases de datos, wrappers de modelos de IA, LLMs centrales y ensambla 
    la máquina de estados finitos que compone las etapas de razonamiento (verificación de dominio, 
    análisis visual, recuperación híbrida transmodal y generación).
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str):
        """
        Inicializa las instancias de infraestructura, pero no monta la red ni los pesos 
        hasta que se llama explícitamente a `initialize()`.

        Args:
            neo4j_uri (str): URI del endpoint de Neo4j (ej. bolt://...).
            neo4j_user (str): Usuario de conexión a la base de datos.
            neo4j_pass (str): Contraseña segura para el grafo.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Inicializando asistente en: {self.device.upper()}")

        self.db = Neo4jClient(neo4j_uri, neo4j_user, neo4j_pass)
        
        self.llm = None
        self.embeddings = None
        
        self.vision_uni = UniWrapper(self.device)
        self.vision_plip = PlipWrapper(self.device)
        
        self.clasificador = None
        self.memory = None
        self.graph = None

    async def initialize(self):
        """
        Lleva a cabo las inicializaciones bloqueantes asíncronas de la app:
        - Confirma la red de Neo4j y crea sus constraints (índices vectoriales multimodales).
        - Hace precarga en VRAM de los modelos HuggingFace / Transformers.
        - Construye y compila el flujo topológico dirigido `StateGraph`.
        """
        await self.db.connect()
        self._init_modelos()
        
        await self.db.crear_esquema()
        import json
        try:
            with open("temario_histologia.json", "r", encoding="utf-8") as f:
                temario = json.load(f)
        except Exception:
            temario = []

        self.clasificador = ClasificadorSemantico(self.llm, self.embeddings, self.device, temario)
        self.memory = SemanticMemory(self.llm, self.embeddings)
        
        self._construir_grafo()

    def _init_modelos(self):
        """
        Provisiona el LLM principal a través de Groq y carga localmente embeddings 
        (SentenceTransformers) y modelos de dominio médico avanzado (UNI, PLIP).
        """
        print("🧠 Inicializando LLM Central (Groq: Llama-4-17B)...")
        if not userdata.get("GROQ_API_KEY"):
            raise ValueError("🚨 FALTA GROQ_API_KEY en las variables de entorno.")
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            api_key=userdata.get("GROQ_API_KEY")
        )
        
        print("🧠 Inicializando Embeddings de Texto Local...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )

        self.vision_uni.load()
        self.vision_plip.load()
        print("✅ Modelos base listos.")

    def _construir_grafo(self):
        """
        Configura los nodos sintéticos (etapas) del flujo LangGraph y traza 
        aristas (edges) condicionadas conformando la lógica y seguridad del chatbot.
        Se determina una ruta alternativa si el `verificar_dominio` detecta spam.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("verificar_dominio", self.nodo_verificar_dominio)
        workflow.add_node("analisis_visual", self.nodo_analisis_visual)
        workflow.add_node("recuperar_memoria", self.nodo_recuperar_memoria)
        workflow.add_node("recuperar_neo4j", self.nodo_recuperar_neo4j)
        workflow.add_node("generar_respuesta", self.nodo_generar_respuesta)
        workflow.add_node("respuesta_fuera_dominio", self.nodo_respuesta_fuera_dominio)

        workflow.set_entry_point("verificar_dominio")
        workflow.add_conditional_edges(
            "verificar_dominio",
            lambda s: "analisis_visual" if s.get("imagen_usuario") else ("recuperar_memoria" if s["clasificacion_dominio"]["valido"] else "respuesta_fuera_dominio"),
            {
                "analisis_visual": "analisis_visual",
                "recuperar_memoria": "recuperar_memoria",
                "respuesta_fuera_dominio": "respuesta_fuera_dominio"
            }
        )
        workflow.add_edge("analisis_visual", "recuperar_memoria")
        workflow.add_edge("recuperar_memoria", "recuperar_neo4j")
        workflow.add_edge("recuperar_neo4j", "generar_respuesta")
        workflow.add_edge("generar_respuesta", END)
        workflow.add_edge("respuesta_fuera_dominio", END)

        self.graph = workflow.compile()
        print("✅ LangGraph compilado (v4.2)")

    @traceable(run_type="chain", name="histologia_rag_verificar_dominio")
    async def nodo_verificar_dominio(self, state: AgentState) -> AgentState:
        """
        Nodo 1 [Filtro]. Criba semántica de contexto.
        Usa simulación heurística para verificar si la pregunta cumple con parámetros histológicos.
        Retorna la validez alterando el diagrama de ejecución subsecuente.
        """
        print("   🔍 [Nodo] Verificando dominio...")
        cls_res = await self.clasificador.clasificar(state["input"], imagen_activa=bool(state.get("imagen_usuario")))
        return {"clasificacion_dominio": cls_res}

    @traceable(run_type="chain", name="histologia_rag_analisis_visual")
    async def nodo_analisis_visual(self, state: AgentState) -> AgentState:
        """
        Nodo 2 [Multimodal]. Expansión de imagen cruda.
        Se usa el modelo Groq para interpretar contextualmente una foto del usuario si es subida.
        La lectura producida es almacenada puramente en forma textual para indexar al motor de búsqueda gráfico.
        """
        print("   👁️ [Nodo] Procesando imagen del usuario...")
        img = state["imagen_usuario"]
        if not img: return {"analisis_visual": "Sin imagen"}
        prompt = f"""Eres un patólogo experto. Describe esta imagen histológica en detalle. 
        Menciona características celulares, tinciones probables (ej. H&E), estructuras visibles y posibles diagnósticos/tejidos. 
        Pregunta del usuario vinculada: {state['input']}"""
        import base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        msgs = [HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])]
        try:
            res = await invoke_con_reintento(self.llm, msgs)
            return {"analisis_visual": res.content}
        except Exception as e:
            return {"analisis_visual": f"Error en análisis visual: {e}"}

    @traceable(run_type="chain", name="histologia_rag_recuperar_memoria")
    async def nodo_recuperar_memoria(self, state: AgentState) -> AgentState:
        """
        Nodo 3 [Retención]. Despertar recuerdos de Qdrant.
        Busca historial similar anterior con el fin de recordar temas en desarrollo y
        retornar esta cadena como input para la generación.
        """
        print("   🧠 [Nodo] Recuperando contexto de memoria...")
        memoria = self.memory.get_context(state["input"]) # Renombrado de retrieve_context a get_context
        return {"memoria_relevante": memoria}

    @traceable(run_type="chain", name="histologia_rag_recuperar_neo4j")
    async def nodo_recuperar_neo4j(self, state: AgentState) -> AgentState:
        """
        Nodo 4 [Grafos vectoriales]. RAG Extendido.
        Lanza embbedings a la búsqueda paralela (texto, imagen_PLIP, imagen_UNI, y match estático de Cypher).
        Conforma la bibliografía en forma de diccionarios adjuntados al input de la LLM.
        """
        print("   🔎 [Nodo] Ejecutando búsqueda híbrida multispace...")
        query = state["input"]
        if state.get("analisis_visual"):
            query += f" (Imagen descrita como: {state['analisis_visual'][:300]})"
            
        emb_query = self.embeddings.embed_query(query)
        from extractors.text import ExtractorEntidades
        extractor = ExtractorEntidades(self.llm)
        entidades = extractor.extraer_de_texto_sync(query)
        
        emb_uni_usr, emb_plip_usr = None, None
        if state.get("imagen_usuario"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                state["imagen_usuario"].convert('RGB').save(tmp.name)
                tmp_path = tmp.name
            emb_uni_usr = self.vision_uni.embed_image(tmp_path)
            emb_plip_usr = self.vision_plip.embed_image(tmp_path)
            os.remove(tmp_path)
            
        resultados = await self.db.busqueda_hibrida(emb_query, emb_uni_usr, emb_plip_usr, entidades, top_k=8)
        return {"contexto_db": resultados}

    @traceable(run_type="chain", name="histologia_rag_generar_respuesta")
    async def nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        """
        Nodo 5 [Conclusión Analítica]. El oráculo RAG.
        Enmascara las referencias y emite un juicio forense empleando toda la información ensamblada 
        (Memoria Semántica, Contexto DB textual y Vectorial). 
        Además adjunta la interaccion de retroalimentación a la base Qdrant.
        """
        print("   ✍️ [Nodo] Generando respuesta final (Rol: Juez Escéptico)...")
        contexto_textos = []
        contexto_imagenes = []
        for c in state["contexto_db"]:
            tipo = c.get("tipo", "texto")
            sim = c.get("similitud", 0)
            fuente = c.get("fuente", "Desconocida")
            txt = c.get("texto", "")
            if tipo == "texto" and txt:
                contexto_textos.append(f"[Fuente: {fuente} | Sim: {sim:.2f}] {txt}")
            elif tipo == "imagen":
                path = c.get("imagen_path", "")
                contexto_imagenes.append(f"[Ref Imagen: {fuente} | Sim: {sim:.2f} | Path: {path}] Leyenda/OCR: {txt}")

        ctx_db_str = "--- TEXTOS DEL MANUAL ---\n" + "\n\n".join(contexto_textos) + "\n\n--- IMÁGENES DE REFERENCIA EN DB ---\n" + "\n\n".join(contexto_imagenes)

        prompt = f"""Eres un médico patólogo experto respondiendo preguntas de histología usando estrictamente los datos proporcionados.
        Si la pregunta requiere identificar una imagen subida por el usuario, adopta el rol de un JUEZ ESCÉPTICO:
        1. Compara minuciosamente el 'Análisis Visual' de la imagen subida con las 'Imágenes de Referencia en DB'.
        2. No asumas que la imagen subida es idéntica a la referencia solo porque la similitud matemática es alta. Busca diferencias biológicas.

        MEMORIA SEMÁNTICA (Contexto previo de la charla):
        {state.get("memoria_relevante", "Sin contexto.")}

        DATOS DEL GRAFO (Neo4j):
        {ctx_db_str}

        ANÁLISIS VÍSUAL DE IMAGEN SUBIDA POR EL USUARIO:
        {state.get("analisis_visual", "El usuario no subió ninguna imagen.")}

        PREGUNTA DEL USUARIO: {state["input"]}

        RESPUESTA FINAL (Nombra las fuentes usadas):"""

        try:
            resp = await invoke_con_reintento(self.llm, [SystemMessage(content="Eres un asistente RAG experto en histología."), HumanMessage(content=prompt)])
            await self.memory.add_interaction(state["input"], resp.content)
            return {"output": resp.content}
        except Exception as e:
            return {"output": f"🚨 Ha ocurrido un error en la generación: {str(e)}"}

    @traceable(run_type="chain", name="histologia_rag_fuera_dominio")
    async def nodo_respuesta_fuera_dominio(self, state: AgentState) -> AgentState:
        """
        NODO DE ERROR (Exception Handler del LangGraph).
        Es desencadenado condicionalmente sólo por preguntas no-biológicas.
        """
        motivo = state["clasificacion_dominio"].get("motivo", "")
        return {"output": f"Lo siento, solo puedo responder preguntas sobre histología (Motivo: {motivo}). ¿Hay algo del manual que quieras explorar?"}

    async def consultar(self, pregunta: str, imagen: Optional[Image.Image] = None, history: List[BaseMessage] = None) -> str:
        """
        Función orquestadora (Entrada final del cliente).
        Desencadena la activación del StateGraph construido al principio y escupe 
        directamente la respuesta final limpia que se manda a pantalla de `cli.py`.

        Args:
             pregunta (str): Input query del usuario.
             imagen (Optional[Image.Image]): Frame cargado si hay archivo.
             history (List[BaseMessage]): Cadena estandarizada de langchain para chat prompt.

        Returns:
             str: Dictado o respuesta patológica forense final.
        """
        if not self.graph:
            return "El agente no está inicializado."
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            initial_state = {
                "input": pregunta,
                "chat_history": history or [],
                "imagen_usuario": imagen,
                "analisis_visual": None,
                "contexto_db": [],
                "memoria_relevante": "",
                "clasificacion_dominio": {"valido": False},
                "output": ""
            }
            final_state = await self.graph.ainvoke(initial_state)
            return final_state["output"]
