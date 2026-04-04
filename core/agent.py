import os
import io
from typing import TypedDict, Annotated, List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, END

# Utils & Config
from utils.config import userdata, _safe, traceable
from utils.resilience import invoke_with_retry
from utils.prompt_loader import load_prompt
# Database & Memory
from db.neo4j_client import Neo4jClient
from db.memory import SemanticMemory
# Classifiers & Vision
from models.classifiers import SemanticClassifier
from models.vision import PlipWrapper, UniWrapper
from extractors.text import EntityExtractor

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

class AgentState(TypedDict):
    """
    Global state payload flowing through the LangGraph architecture.
    """
    input: str
    session_id: str
    chat_history: List[BaseMessage]
    user_image: Optional[Image.Image]  # Retaining UI payload keys for compatibility
    visual_analysis: Optional[str]
    db_context: List[Dict]
    relevant_memory: str
    domain_classification: Dict[str, Any]
    output: str
    trajectory: List[Dict]
    identified_structure: Optional[str]
    recovered_images: List[str]

class Neo4jHistologyAgent:
    """
    Core RAG System Orchestrator built on LangGraph.
    
    Manages database connections, vision model wrappers, central LLMs, and assembles
    the finite state machine for reasoning operations (domain verification,
    visual analysis, hybrid transmodal retrieval, and generation).
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str):
        """
        Initializes infrastructure instances without mounting the network or heavy weights 
        until `initialize()` is explicitly invoked.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"⚠️ Incompatible GPU detected (sm_{cap[0]}{cap[1]}). Forcing CPU fallback.")
                    self.device = "cpu"
            except:
                pass
        print(f"🖥️ Initializing agent on: {self.device.upper()}")

        self.db = Neo4jClient(neo4j_uri, neo4j_user, neo4j_pass)
        
        self.llm = None
        self.embeddings = None
        
        self.vision_uni = UniWrapper(self.device)
        self.vision_plip = PlipWrapper(self.device)
        
        self.classifier = None
        self.memory = None
        self.graph = None

    async def initialize(self):
        """
        Performs asynchronous blocking bootups:
        - Network confirmations and constraint creation for Neo4j.
        - Preloading HuggingFace Transformer models into VRAM.
        - Building and compiling the directed `StateGraph`.
        """
        await self.db.connect()
        self._init_models()
        
        await self.db.create_schema()
        import json
        try:
            with open("temario_histologia.json", "r", encoding="utf-8") as f:
                syllabus = json.load(f)
        except Exception:
            syllabus = []

        self.classifier = SemanticClassifier(self.llm, self.embeddings, self.device, syllabus)
        self.memory = SemanticMemory(self.llm, self.embeddings)
        
        self._build_graph()

    def _init_models(self):
        """
        Provisions the main LLM (Groq) and locally mounts embeddings 
        along side advanced medical domain models (UNI, PLIP).
        """
        print("🧠 Initializing Central LLM (Groq: Llama-4-17B)...")
        if not userdata.get("GROQ_API_KEY"):
            raise ValueError("🚨 MISSING GROQ_API_KEY in environment variables.")
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            api_key=userdata.get("GROQ_API_KEY")
        )
        
        print("🧠 Initializing Local Text Embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )

        self.vision_uni.load()
        self.vision_plip.load()
        print("✅ Base models ready.")

    def _build_graph(self):
        """
        Configures LangGraph synthetic nodes and traces conditional edges.
        Determines fallback routing if `verify_domain` identifies non-biological spam.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("verify_domain", self._node_verify_domain)
        workflow.add_node("visual_analysis", self._node_visual_analysis)
        workflow.add_node("retrieve_memory", self._node_retrieve_memory)
        workflow.add_node("retrieve_neo4j", self._node_retrieve_neo4j)
        workflow.add_node("generate_response", self._node_generate_response)
        workflow.add_node("out_of_domain", self._node_out_of_domain_response)

        workflow.set_entry_point("verify_domain")
        workflow.add_conditional_edges(
            "verify_domain",
            lambda s: "visual_analysis" if s.get("user_image") else ("retrieve_memory" if s["domain_classification"]["valido"] else "out_of_domain"),
            {
                "visual_analysis": "visual_analysis",
                "retrieve_memory": "retrieve_memory",
                "out_of_domain": "out_of_domain"
            }
        )
        workflow.add_edge("visual_analysis", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "retrieve_neo4j")
        workflow.add_edge("retrieve_neo4j", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("out_of_domain", END)

        self.graph = workflow.compile()
        print("✅ LangGraph compiled (v4.4)")

    @traceable(run_type="chain", name="histology_rag_verify_domain")
    async def _node_verify_domain(self, state: AgentState) -> AgentState:
        """
        Node 1 [Filter]. Semantic context screening.
        """
        print("   🔍 [Node] Verifying domain...")
        cls_res = await self.classifier.classify(state["input"], active_image=bool(state.get("user_image")))
        return {"domain_classification": cls_res}

    @traceable(run_type="chain", name="histology_rag_visual_analysis")
    async def _node_visual_analysis(self, state: AgentState) -> AgentState:
        """
        Node 2 [Multimodal]. Raw Image Expansion.
        """
        print("   👁️ [Node] Processing user image...")
        img = state["user_image"]
        if not img: return {"visual_analysis": "Sin imagen"}
        
        raw_prompt = load_prompt("vision_analysis.txt")
        prompt     = raw_prompt.format(INPUT=state['input'])
        
        import base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        msgs = [HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])]
        
        try:
            res = await invoke_with_retry(self.llm, msgs)
            content = res.content
            structure = None
            import re
            match = re.search(r"ESTRUCTURA:\s*(.+)", content, re.IGNORECASE)
            if match:
                structure = match.group(1).strip()
            return {"visual_analysis": content, "identified_structure": structure}
        except Exception as e:
            return {"visual_analysis": f"Visual analysis error: {e}"}

    @traceable(run_type="chain", name="histology_rag_retrieve_memory")
    async def _node_retrieve_memory(self, state: AgentState) -> AgentState:
        """
        Node 3 [Retention]. Wake up Qdrant semantic contexts.
        """
        print("   🧠 [Node] Retrieving memory context...")
        memoria = self.memory.get_context(state["session_id"], state["input"])
        return {"relevant_memory": memoria}

    @traceable(run_type="chain", name="histology_rag_retrieve_neo4j")
    async def _node_retrieve_neo4j(self, state: AgentState) -> AgentState:
        """
        Node 4 [Vector Graphs]. Extended RAG.
        Dispatches embeddings to parallel DB searches (Text, Image PLIP, Image UNI, and exact Cypher match).
        """
        print("   🔎 [Node] Executing hybrid multispace search...")
        query = state["input"]
        if state.get("visual_analysis"):
            query += f" (Imagen descrita como: {state['visual_analysis'][:300]})"
            
        emb_query = self.embeddings.embed_query(query)
        extractor = EntityExtractor(self.llm)
        entidades = await extractor.extract_from_text(query)
        
        emb_uni_usr, emb_plip_usr = None, None
        if state.get("user_image"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                state["user_image"].convert('RGB').save(tmp.name)
                tmp_path = tmp.name
            emb_uni_usr = self.vision_uni.embed_image(tmp_path).tolist()
            emb_plip_usr = self.vision_plip.embed_image(tmp_path).tolist()
            os.remove(tmp_path)
            
        results = await self.db.hybrid_search(emb_query, emb_uni_usr, emb_plip_usr, entidades, top_k=8)
        
        recover_imgs = []
        for r in results:
            if r.get("type") == "imagen" and r.get("image_path"):
                recover_imgs.append(os.path.basename(r.get("image_path")))
                
        return {"db_context": results, "recovered_images": list(set(recover_imgs))}

    @traceable(run_type="chain", name="histology_rag_generate_response")
    async def _node_generate_response(self, state: AgentState) -> AgentState:
        """
        Node 5 [Analytical Conclusion]. The RAG Oracle.
        """
        print("   ✍️ [Node] Generating final response (Role: Sceptical Judge)...")
        context_texts = []
        context_images = []
        for c in state["db_context"]:
            c_type = c.get("type", "texto")
            sim = c.get("similarity", 0)
            source = c.get("source", "Desconocida")
            txt = c.get("text", "")
            if c_type == "texto" and txt:
                context_texts.append(f"[Fuente: {source} | Similitud: {sim:.2f}] {txt}")
            elif c_type == "imagen":
                path = c.get("image_path", "")
                context_images.append(f"[Imagen Ref: {source} | Similitud: {sim:.2f} | Ruta: {path}] Leyenda/OCR: {txt}")

        ctx_db_str = "--- TEXTOS DEL MANUAL ---\n" + "\n\n".join(context_texts) + "\n\n--- IMÁGENES DE REFERENCIA EN DB ---\n" + "\n\n".join(context_images)

        raw_prompt = load_prompt("rag_generation.txt")
        prompt     = raw_prompt.format(
            MEMORY=state.get("relevant_memory", "Sin contexto."),
            DB_CONTEXT=ctx_db_str,
            VISUAL_ANALYSIS=state.get("visual_analysis", "El usuario no subió ninguna imagen."),
            INPUT=state["input"]
        )

        try:
            resp = await invoke_with_retry(self.llm, [SystemMessage(content="Eres un asistente RAG experto en histología."), HumanMessage(content=prompt)])
            await self.memory.add_interaction(state["session_id"], state["input"], resp.content)
            return {"output": resp.content}
        except Exception as e:
            return {"output": f"🚨 Error generation block: {str(e)}"}

    @traceable(run_type="chain", name="histology_rag_out_of_domain")
    async def _node_out_of_domain_response(self, state: AgentState) -> AgentState:
        """
        ERROR NODE. Exception Handler conditionally triggered by off-topic biological queries.
        """
        reason = state["domain_classification"].get("motivo", "")
        return {"output": f"Lo siento, solo puedo responder preguntas sobre histología (Motivo: {reason}). ¿Hay algo del manual que quieras explorar?"}

    async def query(self, text_query: str, session_id: str = "default_session", image: Optional[Image.Image] = None, history: List[BaseMessage] = None) -> Dict[str, Any]:
        """
        Primary Orchestrator Entrypoint wrapper replacing old consultar().
        """
        if not self.graph:
            return {"respuesta": "Agent not properly initialized."}
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            import time
            start_t = time.time()
            initial_state = {
                "input": text_query,
                "session_id": session_id,
                "chat_history": history or [],
                "user_image": image,
                "visual_analysis": None,
                "db_context": [],
                "relevant_memory": "",
                "domain_classification": {"valido": False},
                "output": "",
                "trajectory": [],
                "identified_structure": None,
                "recovered_images": []
            }
            final_state = await self.graph.ainvoke(initial_state)
            time_elapsed = round(time.time() - start_t, 2)
            
            trajectory = [{"nodo": "GenerarRespuesta", "tiempo_total": time_elapsed}]
            
            return {
                "answer": final_state["output"],
                "trajectory": trajectory,
                "identified_structure": final_state.get("identified_structure"),
                "recovered_images": final_state.get("recovered_images", [])
            }
