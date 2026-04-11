# =============================================================================
# RAG Multimodal de Histología con ImageBind + Neo4j — VERSIÓN 4.2
# =============================================================================
# Cambios sobre v4.1:
#   1. REFACTORING CONSULTAS DE TEXTO: Flujo bifurcado en LangGraph.
#      Consultas sin imagen saltan procesar_imagen y analisis_comparativo.
#   2. ROUTER CONDICIONAL: _route_por_modo decide el camino del grafo.
#   3. UMBRALES DIFERENCIADOS: texto puro usa umbral 0.45 (más permisivo),
#      modo imagen mantiene 0.6 para texto y 0.45 para imágenes.
#   4. SYSTEM PROMPT DIFERENCIADO: modo texto tiene prompt optimizado para
#      respuestas enciclopédicas sin referencias a imágenes del usuario.
#   5. MEJOR MANEJO DE NO-CONTEXTO: en modo texto, mensaje amable indicando
#      que no se encontró info, en vez de "fuera de dominio".
# =============================================================================
# Cambios v4.1:
#   1. RAGAS eliminado completamente.
#   2. MEMORIA DE IMAGEN PERSISTENTE.
#   3. CLASIFICADOR SEMÁNTICO.
#   4. MEMORIA SIEMPRE ACTUALIZADA.
#   5. MODO INTERACTIVO MEJORADO.
# =============================================================================

import os
import json
import time
import asyncio
import nest_asyncio
import torch
import numpy as np
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from PIL import Image
import base64
import glob
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# UNI & PLIP
import timm
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import login

# Verificar HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Logueado en Hugging Face")
    except Exception as e:
        print(f"⚠️ Error login HF: {e}")
else:
    print("⚠️ HF_TOKEN no encontrado en .env (necesario para UNI)")

# Neo4j
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable

import fitz # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Wrapper para leer variables de entorno (compatible con .env)
class userdata:
    @staticmethod
    def get(key):
        return os.environ.get(key)

nest_asyncio.apply()

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
# CONFIGURACIÓN GLOBAL
SIMILARITY_THRESHOLD  = 0.45
# Dimensiones de embeddings
DIM_TEXTO        = 384
DIM_IMG_UNI      = 1024
DIM_IMG_PLIP     = 512

DIRECTORIO_IMAGENES   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes_extraidas")
DIRECTORIO_PDFS       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")

# Índices Neo4j
INDEX_TEXTO = "histo_text"      # Text Vector
INDEX_UNI   = "histo_img_uni"   # UNI Image
INDEX_PLIP  = "histo_img_plip"  # PLIP Image

NEO4J_GRAPH_DEPTH     = 2
SIMILAR_IMG_THRESHOLD = 0.85 # Más alto para modelos especializados

FEATURES_DISCRIMINATORIAS = [
    "presencia/ausencia de lumen central",
    "estratificación celular (capas concéntricas vs difusa)",
    "tipo de queratinización (parakeratosis, ortoqueratosis, ninguna)",
    "aspecto del núcleo (picnótico, fantasma, ausente, vesicular)",
    "células fantasma (sí/no)",
    "material amorfo central (sí/no y aspecto)",
    "patrón de tinción H&E (eosinofilia, basofilia)",
    "tamaño estimado de la estructura",
    "tejido circundante (estroma, epitelio, piel, otro)",
    "reacción inflamatoria perilesional (sí/no, tipo)",
]

# Anclas semánticas para el clasificador de dominio
ANCLAS_SEMANTICAS_HISTOLOGIA = [
    "histología tejido celular microscopía",
    "tipos de tejido epitelial conectivo muscular nervioso",
    "coloración hematoxilina eosina H&E tinción histológica",
    "estructuras celulares núcleo citoplasma membrana",
    "diagnóstico diferencial patología biopsia",
    "glándulas epitelio estratificado cilíndrico simple",
    "identificar tejido muestra microscópica",
    "¿qué tipo de tejido es este?",
    "¿cuál es la estructura observada en la imagen?",
    "clasificar célula estructura histológica",
    "tumor quiste folículo cuerpo lúteo albicans",
    "corte histológico preparación muestra lámina",
    "tejido epitelial simple cilíndrico estratificado pseudoestratificado",
    "tejido conectivo laxo denso adiposo cartilaginoso óseo",
    "tejido muscular liso estriado cardíaco esquelético",
    "tejido nervioso neurona glía axón dendrita",
]

def _safe(value, default: str = "") -> str:
    return value if isinstance(value, str) and value else default

async def invoke_con_reintento(llm, messages, max_retries=5):
    import asyncio
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API/Servidor Ocupado (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    await asyncio.sleep(espera)
                else:
                    raise e
            else:
                raise e

def invoke_con_reintento_sync(llm, messages, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API/Servidor Ocupado (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e

def embed_query_con_reintento(embeddings, texto: str, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return embeddings.embed_query(texto)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API en embeddings (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e

def embed_documents_con_reintento(embeddings, textos: list, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(textos)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ Límite de cuota API en embeddings (429/503) - reintentando en {espera}s... (Intento {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e


# =============================================================================
# LANGSMITH
# =============================================================================

def setup_langsmith_environment():
    config = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY":    userdata.get("LANGSMITH_API_KEY"),
        "LANGCHAIN_ENDPOINT":   "https://api.smith.langchain.com",
        "LANGCHAIN_PROJECT":    "rag_histologia_neo4j_v4"
    }
    for key, value in config.items():
        if value:
            os.environ[key] = value
    try:
        from langsmith import traceable, Client
        client = Client()
        print(f"✅ LangSmith — Proyecto: {os.environ.get('LANGCHAIN_PROJECT')}")
        return True, traceable, client
    except Exception as e:
        print(f"⚠️ LangSmith no disponible: {e}")
        def dummy_traceable(*args, **kwargs):
            def decorator(func): return func
            if len(args) == 1 and callable(args[0]): return args[0]
            return decorator
        return False, dummy_traceable, None

LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith_environment()


# =============================================================================
# WRAPPERS DE MODELOS (PLIP & UNI)
# =============================================================================

class PlipWrapper:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        print("🔄 Cargando PLIP (vinid/plip)...")
        try:
            self.model = CLIPModel.from_pretrained("vinid/plip").to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained("vinid/plip")
            print("✅ PLIP cargado")
        except Exception as e:
            print(f"❌ Error cargando PLIP: {e}")

    def embed_image(self, image_path: str) -> np.ndarray:
        if not self.model: return np.zeros(DIM_IMG_PLIP)
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                pooled = vision_out.pooler_output
                image_features = self.model.visual_projection(pooled)  # [1, 512]
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding PLIP: {e}")
            return np.zeros(DIM_IMG_PLIP)

class UniWrapper:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.transform = None

    def load(self):
        print("🔄 Cargando UNI (MahmoodLab)...")
        try:
            # UNI usa timm con hf_hub
            self.model = timm.create_model(
                "hf_hub:MahmoodLab/UNI", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=True
            )
            self.model.to(self.device).eval()
            
            # Transformación estándar de UNI (ViT-L/16)
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
            self.transform = create_transform(**config)
            print("✅ UNI cargado")
        except Exception as e:
            print(f"❌ Error cargando UNI: {e}")

    def embed_image(self, image_path: str) -> np.ndarray:
        if not self.model: return np.zeros(DIM_IMG_UNI)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = self.model(image_tensor) # UNI returns raw features [1, 1024]
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding UNI: {e}")
            return np.zeros(DIM_IMG_UNI)


# =============================================================================
# CLIENTE NEO4J (Adaptado para 3 índices)
# =============================================================================

class Neo4jClient:
    """
    Wrapper async para Neo4j que encapsula:
    - Conexión (AuraDB cloud o local)
    - Creación del esquema de grafo y vector index
    - Operaciones de escritura (indexación)
    - Operaciones de lectura (búsqueda híbrida)
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri      = uri
        self.user     = user
        self.password = password
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        await self._driver.verify_connectivity()
        print(f"✅ Neo4j conectado: {self.uri}")

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def run(self, query: str, params: Dict = None) -> List[Dict]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) for record in await result.data()]

    async def crear_esquema(self):
        print("🏗️ Creando esquema Neo4j (v4.2 UNI + PLIP)...")
        constraints = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT imagen_id IF NOT EXISTS FOR (i:Imagen) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT pdf_nombre IF NOT EXISTS FOR (p:PDF) REQUIRE p.nombre IS UNIQUE",
            "CREATE CONSTRAINT tejido_nombre IF NOT EXISTS FOR (t:Tejido) REQUIRE t.nombre IS UNIQUE",
            "CREATE CONSTRAINT estructura_nombre IF NOT EXISTS FOR (e:Estructura) REQUIRE e.nombre IS UNIQUE",
            "CREATE CONSTRAINT tincion_nombre IF NOT EXISTS FOR (t:Tincion) REQUIRE t.nombre IS UNIQUE",
        ]
        for c in constraints:
            try:
                await self.run(c)
            except Exception as e:
                print(f"  ⚠️ Constraint: {e}")

        # --- Validación de Índices Vectoriales (Dimensionalidad) ---
        try:
            indexes = await self.run("SHOW INDEXES YIELD name, type, options")
            for idx in indexes:
                if idx["type"] == "VECTOR":
                    name = idx["name"]
                    config = idx["options"].get("indexConfig", {})
                    dims = config.get("vector.dimensions")
                    
                    target_dims = None
                    if name == INDEX_TEXTO: target_dims = DIM_TEXTO
                    elif name == INDEX_UNI: target_dims = DIM_IMG_UNI
                    elif name == INDEX_PLIP: target_dims = DIM_IMG_PLIP
                    
                    if target_dims and dims != target_dims:
                        print(f"  ⚠️ Dimensión incorrecta en índice Neo4j '{name}' ({dims} != {target_dims}). Recreando...")
                        await self.run(f"DROP INDEX {name}")
        except Exception as e:
            print(f"  ⚠️ Error validando índices Neo4j: {e}")

        # 3 Índices Vectoriales
        vector_queries = [
            # 1. TEXTO
            f"""
            CREATE VECTOR INDEX {INDEX_TEXTO} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {DIM_TEXTO},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            # 2. IMAGEN UNI
            f"""
            CREATE VECTOR INDEX {INDEX_UNI} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_uni
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {DIM_IMG_UNI},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            # 3. IMAGEN PLIP
            f"""
            CREATE VECTOR INDEX {INDEX_PLIP} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_plip
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {DIM_IMG_PLIP},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        for vq in vector_queries:
            try:
                await self.run(vq)
            except Exception as e:
                print(f"  ⚠️ Vector index error: {e}")

        print("✅ Esquema Neo4j listo (3 índices)")

    async def upsert_pdf(self, nombre: str):
        await self.run("MERGE (p:PDF {nombre: $nombre})", {"nombre": nombre})

    async def upsert_chunk(self, chunk_id: str, texto: str, fuente: str,
                            chunk_idx: int, embedding: List[float],
                            entidades: Dict[str, List[str]]):
        await self.run("""
            MERGE (c:Chunk {id: $id})
            SET c.texto = $texto, c.fuente = $fuente,
                c.chunk_id = $chunk_idx, c.embedding = $embedding
            WITH c
            MERGE (pdf:PDF {nombre: $fuente})
            MERGE (c)-[:PERTENECE_A]->(pdf)
        """, {
            "id": chunk_id, "texto": texto, "fuente": fuente,
            "chunk_idx": chunk_idx, "embedding": embedding
        })
        for tejido in entidades.get("tejidos", []):
            await self.run("""
                MERGE (t:Tejido {nombre: $nombre})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"nombre": tejido, "chunk_id": chunk_id})
        for estructura in entidades.get("estructuras", []):
            await self.run("""
                MERGE (e:Estructura {nombre: $nombre})
                WITH e MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
            """, {"nombre": estructura, "chunk_id": chunk_id})
        for tincion in entidades.get("tinciones", []):
            await self.run("""
                MERGE (t:Tincion {nombre: $nombre})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"nombre": tincion, "chunk_id": chunk_id})
        for tejido in entidades.get("tejidos", []):
            for estructura in entidades.get("estructuras", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tejido})
                    MERGE (e:Estructura {nombre: $estructura})
                    MERGE (t)-[:CONTIENE]->(e)
                """, {"tejido": tejido, "estructura": estructura})
            for tincion in entidades.get("tinciones", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tejido})
                    MERGE (ti:Tincion {nombre: $tincion})
                    MERGE (t)-[:TENIDA_CON]->(ti)
                """, {"tejido": tejido, "tincion": tincion})
        for estructura in entidades.get("estructuras", []):
            for tincion in entidades.get("tinciones", []):
                await self.run("""
                    MERGE (e:Estructura {nombre: $estructura})
                    MERGE (ti:Tincion {nombre: $tincion})
                    MERGE (e)-[:TENIDA_CON]->(ti)
                """, {"estructura": estructura, "tincion": tincion})

    async def upsert_imagen(self, imagen_id: str, path: str, fuente: str,
                             pagina: int, ocr_text: str, texto_pagina: str,
                             emb_uni: List[float], emb_plip: List[float],
                             caption: str = ""):
        await self.run("""
            MERGE (i:Imagen {id: $id})
            SET i.path = $path, i.fuente = $fuente,
                i.pagina = $pagina, i.ocr_text = $ocr_text,
                i.texto_pagina = $texto_pagina,
                i.caption = $caption,
                i.embedding_uni = $emb_uni,
                i.embedding_plip = $emb_plip
            WITH i
            MERGE (pdf:PDF {nombre: $fuente})
            MERGE (i)-[:PERTENECE_A]->(pdf)
            MERGE (pag:Pagina {numero: $pagina, pdf_nombre: $fuente})
            MERGE (i)-[:EN_PAGINA]->(pag)
        """, {
            "id": imagen_id, "path": path, "fuente": fuente,
            "pagina": pagina, "ocr_text": ocr_text, "texto_pagina": texto_pagina,
            "caption": caption,
            "emb_uni": emb_uni, "emb_plip": emb_plip
        })

    async def crear_relaciones_similitud(self, umbral: float = SIMILAR_IMG_THRESHOLD):
        # Usamos UNI para similitud visual (multimodal)
        print(f"🔗 Creando relaciones :SIMILAR_A (usando UNI, umbral={umbral})...")
        imagenes = await self.run(
            "MATCH (i:Imagen) WHERE i.embedding_uni IS NOT NULL RETURN i.id AS id, i.embedding_uni AS emb"
        )
        if len(imagenes) < 2:
            return
        creadas = 0
        for img in imagenes:
            resultado = await self.run("""
                CALL db.index.vector.queryNodes($index, 6, $emb)
                YIELD node AS vecino, score
                WHERE vecino.id <> $id AND score >= $umbral
                WITH vecino, score
                MATCH (origen:Imagen {id: $id})
                MERGE (origen)-[r:SIMILAR_A]->(vecino)
                SET r.score = score
                RETURN count(*) AS n
            """, {
                "index": INDEX_UNI,
                "emb": img["emb"], "id": img["id"], "umbral": umbral
            })
            creadas += resultado[0]["n"] if resultado else 0
        print(f"✅ {creadas} relaciones :SIMILAR_A creadas")

    async def busqueda_vectorial(self, embedding: List[float],
                                  index_name: str, top_k: int = 10) -> List[Dict]:
        is_text_index = index_name == INDEX_TEXTO
        if is_text_index:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS c, score
                RETURN c.id AS id, c.texto AS texto, c.fuente AS fuente,
                       'texto' AS tipo, null AS imagen_path, score AS similitud
                ORDER BY similitud DESC
            """
        else:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS i, score
                RETURN i.id AS id, 
                       coalesce(
                           CASE WHEN i.caption IS NOT NULL AND i.caption <> '' THEN i.caption END,
                           CASE WHEN i.ocr_text IS NOT NULL AND i.ocr_text <> '' THEN i.ocr_text END,
                           i.texto_pagina
                       ) AS texto, 
                       i.fuente AS fuente,
                       'imagen' AS tipo, i.path AS imagen_path, score AS similitud
                ORDER BY similitud DESC
            """
        try:
            return await self.run(query, {"index": index_name, "emb": embedding, "k": top_k})
        except Exception as e:
            print(f"⚠️ Error búsqueda vectorial {index_name}: {e}")
            return []

    async def busqueda_por_entidades(self, entidades: Dict[str, List[str]],
                                      top_k: int = 10) -> List[Dict]:
        tejidos     = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        tinciones   = entidades.get("tinciones", [])
        if not any([tejidos, estructuras, tinciones]):
            return []

        where_clauses = []
        params: Dict[str, Any] = {}
        if tejidos:
            where_clauses.append("ANY(t IN tejidos WHERE t.nombre IN $tejidos)")
            params["tejidos"] = tejidos
        if estructuras:
            where_clauses.append("ANY(e IN estructuras WHERE e.nombre IN $estructuras)")
            params["estructuras"] = estructuras
        if tinciones:
            where_clauses.append("ANY(ti IN tinciones WHERE ti.nombre IN $tinciones)")
            params["tinciones"] = tinciones

        where_str = " OR ".join(where_clauses)
        query = f"""
            MATCH (c:Chunk)
            OPTIONAL MATCH (c)-[:MENCIONA]->(t:Tejido)
            OPTIONAL MATCH (c)-[:MENCIONA]->(e:Estructura)
            OPTIONAL MATCH (c)-[:MENCIONA]->(ti:Tincion)
            WITH c,
                 collect(DISTINCT t) AS tejidos,
                 collect(DISTINCT e) AS estructuras,
                 collect(DISTINCT ti) AS tinciones
            WHERE {where_str}
            RETURN c.id AS id, c.texto AS texto, c.fuente AS fuente,
                   'texto' AS tipo, null AS imagen_path, 0.49 AS similitud
            LIMIT $top_k
        """
        params["top_k"] = top_k
        try:
            return await self.run(query, params)
        except Exception as e:
            print(f"⚠️ Error búsqueda entidades: {e}")
            return []

    async def expandir_vecindad(self, node_ids: List[str],
                                 depth: int = NEO4J_GRAPH_DEPTH) -> List[Dict]:
        if not node_ids:
            return []
        query = """
            UNWIND $ids AS nid
            MATCH (n {id: nid})

            // Expansión 1: otros nodos en el mismo PDF (excluir el nodo origen)
            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)
            WHERE vecino_pdf.id <> nid
            WITH n, nid, collect(DISTINCT vecino_pdf)[..5] AS list_pdf

            // Expansión 2: chunks que comparten entidades (excluir el nodo origen)
            OPTIONAL MATCH (n)-[:MENCIONA]->(entidad)<-[:MENCIONA]-(vecino_entidad:Chunk)
            WHERE vecino_entidad.id <> nid
            WITH n, nid, list_pdf, collect(DISTINCT vecino_entidad)[..5] AS list_ent

            // Expansión 3: imágenes similares por embedding
            OPTIONAL MATCH (n)-[:SIMILAR_A]-(vecino_similar:Imagen)
            WITH n, nid, list_pdf, list_ent, collect(DISTINCT vecino_similar)[..5] AS list_sim

            // Expansión 4: imágenes de la misma página (Fix #4)
            OPTIONAL MATCH (n)-[:EN_PAGINA]->(pag:Pagina)<-[:EN_PAGINA]-(img_pag:Imagen)
            WHERE img_pag.id <> nid
            WITH n, nid, list_pdf, list_ent, list_sim, collect(DISTINCT img_pag)[..5] AS list_pag

            WITH n, $ids AS ids_originales,
                 list_pdf + list_ent + list_sim + list_pag AS vecinos_raw

            UNWIND vecinos_raw AS v

            WITH n, v, ids_originales
            WHERE v IS NOT NULL AND NOT v.id IN ids_originales

            RETURN DISTINCT
                v.id AS id,
                CASE 
                    WHEN v:Imagen THEN coalesce(
                        CASE WHEN v.caption IS NOT NULL AND v.caption <> '' THEN v.caption END,
                        CASE WHEN v.ocr_text IS NOT NULL AND v.ocr_text <> '' THEN v.ocr_text END,
                        v.texto_pagina, ''
                    )
                    ELSE coalesce(v.texto, '') 
                END AS texto,
                v.fuente AS fuente,
                CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS tipo,
                CASE WHEN v:Imagen THEN v.path ELSE null END AS imagen_path,
                CASE 
                    WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
                         (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
                    THEN 0.95 
                    ELSE 0.3 
                END AS similitud
            LIMIT 15
        """
        try:
            return await self.run(query, {"ids": node_ids})
        except Exception as e:
            print(f"⚠️ Error expansión vecindad: {e}")
            return []

    async def busqueda_camino_semantico(self,
                                         tejido_origen: Optional[str],
                                         tejido_destino: Optional[str]) -> List[Dict]:
        if not tejido_origen or not tejido_destino:
            return []
        query = """
            MATCH (origen {nombre: $origen}), (destino {nombre: $destino})
            MATCH path = shortestPath((origen)-[*1..4]-(destino))
            UNWIND nodes(path) AS nodo
            OPTIONAL MATCH (c:Chunk)-[:MENCIONA]->(nodo)
            RETURN DISTINCT c.id AS id, c.texto AS texto, c.fuente AS fuente,
                   'texto' AS tipo, null AS imagen_path, 0.4 AS similitud
            LIMIT 5
        """
        try:
            return await self.run(query, {"origen": tejido_origen, "destino": tejido_destino})
        except Exception as e:
            print(f"⚠️ Error búsqueda camino: {e}")
            return []

    async def busqueda_hibrida(self,
                                texto_embedding: Optional[List[float]],
                                imagen_embedding_uni: Optional[List[float]],
                                imagen_embedding_plip: Optional[List[float]],
                                entidades: Dict[str, List[str]],
                                top_k: int = 10) -> List[Dict]:
        res_texto = []
        res_uni   = []
        res_plip  = []
        res_ent   = []
        res_vec   = []

        # 1. Búsqueda Texto (Gemini)
        if texto_embedding:
            res_texto = await self.busqueda_vectorial(texto_embedding, INDEX_TEXTO, top_k)

        # 2. Búsqueda Imagen UNI
        if imagen_embedding_uni:
            res_uni = await self.busqueda_vectorial(imagen_embedding_uni, INDEX_UNI, top_k)

        # 3. Búsqueda Imagen PLIP
        if imagen_embedding_plip:
            res_plip = await self.busqueda_vectorial(imagen_embedding_plip, INDEX_PLIP, top_k)

        # 4. Entidades
        res_ent = await self.busqueda_por_entidades(entidades, top_k)

        # Vecindad sobre los mejores
        todos = res_texto + res_uni + res_plip
        top_ids = [r["id"] for r in todos[:6] if r.get("id")]
        if top_ids:
            res_vec = await self.expandir_vecindad(top_ids)

        combined: Dict[str, Dict] = {}

        def agregar(resultados: List[Dict], peso: float):
            for r in resultados:
                key = r.get("id") or f"{r.get('fuente')}_{str(r.get('texto',''))[:40]}"
                if not r.get("texto") and not r.get("imagen_path"):
                    continue
                sim_ponderada = r.get("similitud", 0) * peso
                if key not in combined:
                    combined[key] = {**r, "similitud": sim_ponderada}
                else:
                    combined[key]["similitud"] += sim_ponderada

        # Pesos dinámicos según modo de consulta (Fix #5)
        tiene_imagen = imagen_embedding_uni is not None or imagen_embedding_plip is not None
        if tiene_imagen:
            agregar(res_texto, 0.40)  # Texto complementa cuando hay imagen
            agregar(res_uni,   0.70)  # UNI domina visual
            agregar(res_plip,  0.70)  # PLIP domina visual
            agregar(res_ent,   0.60)  # Entidades siempre importantes
        else:
            agregar(res_texto, 0.80)  # Texto domina sin imagen
            agregar(res_uni,   0.20)  # UNI poco relevante sin query visual
            agregar(res_plip,  0.20)  # PLIP poco relevante sin query visual
            agregar(res_ent,   0.60)  # Entidades siempre importantes
        agregar(res_vec,   0.20)

        final = sorted(combined.values(), key=lambda x: x["similitud"], reverse=True)

        print(f"   📊 Híbrida: Txt={len(res_texto)} | "
              f"UNI={len(res_uni)} | PLIP={len(res_plip)} | Ent={len(res_ent)} | Vec={len(res_vec)} -> {len(final)}")

        return final[:15]


# =============================================================================
# MEMORIA SEMÁNTICA CON PERSISTENCIA DE IMAGEN
# =============================================================================

class SemanticMemory:
    """
    Mantiene historial de conversación y la última imagen activa.
    La imagen persiste entre turnos hasta que el usuario la cambie
    explícitamente con el comando 'nueva imagen'.
    Añade persistencia en Qdrant con vectores texto, UNI y PLIP.
    """

    def __init__(self, llm, embeddings=None, uni=None, plip=None, max_entries: int = 10):
        self.llm            = llm
        self.embeddings     = embeddings
        self.uni            = uni
        self.plip           = plip
        self.conversations  = []
        self.max_entries    = max_entries
        self.summary        = ""
        self.direct_history = ""

        # Persistencia de imagen entre turnos
        self.imagen_activa_path: Optional[str] = None
        self.imagen_turno_subida: int = 0
        self.turno_actual: int = 0
        
        # Qdrant init
        self.collection_name = "memoria_histo"
        self.qdrant = QdrantClient(path="./qdrant_memoria")
        
        # --- Validación de Colección Qdrant (Dimensionalidad) ---
        try:
            info = self.qdrant.get_collection(self.collection_name)
            config = info.config.params.vectors
            
            # Verificar dimensión de 'texto'
            dims_actual = 0
            if isinstance(config, dict) and "texto" in config:
                dims_actual = config["texto"].size
            
            if dims_actual != DIM_TEXTO:
                print(f"   ⚠️ Dimensión de Qdrant incorrecta ({dims_actual} != {DIM_TEXTO}). Recreando...")
                self.qdrant.delete_collection(self.collection_name)
                raise Exception("Mismatch forcing recreation")
        except Exception:
            print(f"   🗂️ Configurando colección Qdrant '{self.collection_name}'...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "texto": VectorParams(size=DIM_TEXTO, distance=Distance.COSINE),
                    "uni": VectorParams(size=DIM_IMG_UNI, distance=Distance.COSINE),
                    "plip": VectorParams(size=DIM_IMG_PLIP, distance=Distance.COSINE),
                }
            )

    def set_imagen(self, path: Optional[str]):
        """Registra una nueva imagen activa. None = limpiar."""
        if path and os.path.exists(path):
            self.imagen_activa_path  = path
            self.imagen_turno_subida = self.turno_actual
            print(f"   📌 Imagen activa registrada (turno {self.turno_actual}): {path}")
        elif path is None:
            self.imagen_activa_path = None
            print("   🗑️  Imagen activa limpiada")

    def get_imagen_activa(self) -> Optional[str]:
        if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
            return self.imagen_activa_path
        return None

    def tiene_imagen_previa(self) -> bool:
        return self.get_imagen_activa() is not None

    def add_interaction(self, query: str, response: str):
        """Guarda siempre la interacción, independientemente del resultado RAG."""
        self.turno_actual += 1
        self.conversations.append({
            "query":    query,
            "response": response,
            "turno":    self.turno_actual,
            "imagen":   self.imagen_activa_path
        })
        if len(self.conversations) > self.max_entries:
            self.conversations.pop(0)

        self.direct_history += f"\nUsuario: {query}\nAsistente: {response}\n"
        if len(self.conversations) > 3:
            recent = self.conversations[-3:]
            self.direct_history = ""
            for conv in recent:
                img_nota = (f" [con imagen: {os.path.basename(conv['imagen'])}]"
                            if conv.get("imagen") else "")
                self.direct_history += (
                    f"\nUsuario{img_nota}: {conv['query']}\n"
                    f"Asistente: {conv['response']}\n"
                )
        self._update_summary()
        
        # Guardar en Qdrant cada 5 interacciones
        if self.turno_actual % 5 == 0 and len(self.conversations) > 0 and self.embeddings:
            self._guardar_memoria_qdrant()

    def _guardar_memoria_qdrant(self):
        print("   🧠 Generando resumen profundo para guardar en memoria (Qdrant)...")
        try:
            # Resumir contexto reciente
            resp = invoke_con_reintento_sync(self.llm, [
                SystemMessage(content="Genera un resumen detallado y técnico del siguiente historial de conversación sobre histología, destacando las entidades mencionadas y las conclusiones."),
                HumanMessage(content=self.direct_history)
            ])
            resumen = resp.content
            
            # Embeddings del resumen textual
            emb_texto = embed_query_con_reintento(self.embeddings, resumen)
            
            # Embeddings visuales de la imagen activa en este bloque (si la hay)
            emb_uni = [0.0] * DIM_IMG_UNI
            emb_plip = [0.0] * DIM_IMG_PLIP
            if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
                if self.uni:
                    emb_uni = self.uni.embed_image(self.imagen_activa_path).tolist()
                if self.plip:
                    emb_plip = self.plip.embed_image(self.imagen_activa_path).tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "texto": emb_texto,
                    "uni": emb_uni,
                    "plip": emb_plip
                },
                payload={
                    "resumen": resumen,
                    "turno_fin": self.turno_actual,
                    "tiene_imagen": self.imagen_activa_path is not None,
                    "imagen_path": self.imagen_activa_path
                }
            )
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print("   ✅ Resumen guardado en Qdrant (memoria_histo)")
        except Exception as e:
            print(f"   ⚠️ Error guardando memoria en Qdrant: {e}")

    def _update_summary(self):
        try:
            if len(self.conversations) > 6:
                resp = invoke_con_reintento_sync(self.llm, [
                    SystemMessage(content="Resume estas consultas de histología manteniendo términos técnicos:"),
                    HumanMessage(content=self.direct_history)
                ])
                self.summary = f"Resumen: {resp.content}\n\nRecientes:{self.direct_history}"
            else:
                self.summary = f"Recientes:{self.direct_history}"
        except Exception as e:
            self.summary = f"Recientes:{self.direct_history}"

    def get_context(self, query: str = "") -> str:
        ctx = self.summary.strip() or "No hay consultas previas."
        
        # Recuperar de Qdrant
        if query and self.embeddings:
            try:
                emb_query = embed_query_con_reintento(self.embeddings, query)
                resultados = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=emb_query,
                    using="texto",
                    limit=2
                ).points
                
                memorias_recuperadas = [r.payload['resumen'] for r in resultados if r.score > 0.4]
                if memorias_recuperadas:
                    ctx += "\n\n[Memorias históricas recuperadas:]\n" + "\n- ".join(memorias_recuperadas)
            except Exception as e:
                print(f"   ⚠️ Error recuperando memoria Qdrant: {e}")

        if self.imagen_activa_path:
            ctx += (f"\n\n[Imagen activa en el chat: "
                    f"{os.path.basename(self.imagen_activa_path)}]")
        return ctx


# =============================================================================
# CLASIFICADOR SEMÁNTICO — reemplaza verificación por keywords
# =============================================================================

class ClasificadorSemantico:
    """
    Determina si una consulta pertenece al dominio histológico combinando:
      1. Similitud coseno contra la ontología extraída del contenido (temario).
         Fallback a anclas semánticas hardcodeadas si el temario está vacío.
      2. Razonamiento LLM con contexto de imagen disponible.
    """

    UMBRAL_SIMILITUD = 0.45
    UMBRAL_LLM       = 0.49

    def __init__(self, llm, embeddings, device: str, temario: List[str]):
        self.llm        = llm
        self.embeddings = embeddings
        self.device     = device
        self._temario: List[str]              = temario
        self._anclas_emb: Optional[np.ndarray] = None
        self._temario_emb: Optional[np.ndarray] = None

    @property
    def temario(self) -> List[str]:
        return self._temario

    @temario.setter
    def temario(self, value: List[str]):
        self._temario = value
        self._temario_emb = None  # Invalidar cache al actualizar ontología
        print(f"   🔄 Ontología actualizada ({len(value)} temas) — cache de embeddings invalidado")

    def _embed_textos(self, textos: List[str]) -> np.ndarray:
        return np.array(embed_documents_con_reintento(self.embeddings, textos))

    def _get_anclas_emb(self) -> np.ndarray:
        """Embeddings de anclas hardcodeadas (fallback cuando no hay temario)."""
        if self._anclas_emb is None:
            print("   🔄 Precalculando embeddings de anclas semánticas (fallback)...")
            self._anclas_emb = self._embed_textos(ANCLAS_SEMANTICAS_HISTOLOGIA)
        return self._anclas_emb

    def _get_temario_emb(self) -> Optional[np.ndarray]:
        """Embeddings de la ontología real extraída del contenido."""
        if not self._temario:
            return None
        if self._temario_emb is None:
            print(f"   🔄 Precalculando embeddings de ontología ({len(self._temario)} temas)...")
            self._temario_emb = self._embed_textos(self._temario)
        return self._temario_emb

    def similitud_con_dominio(self, consulta: str) -> float:
        try:
            q_emb = np.array(embed_query_con_reintento(self.embeddings, consulta))

            # Priorizar ontología extraída del contenido real
            temario_emb = self._get_temario_emb()
            if temario_emb is not None and len(temario_emb) > 0:
                sims = (q_emb @ temario_emb.T).flatten()
                return float(np.max(sims))

            # Fallback a anclas hardcodeadas (solo antes de indexar)
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
        # Paso 1: Similitud ImageBind
        sim = self.similitud_con_dominio(consulta)
        print(f"   📐 Similitud semántica con dominio histológico: {sim:.4f}")

        umbral_efectivo = self.UMBRAL_SIMILITUD * (0.6 if imagen_activa else 1.0)

        if sim >= umbral_efectivo:
            return {
                "valido":            True,
                "tema_encontrado":   None,
                "motivo":            f"Similitud ImageBind {sim:.3f} ≥ umbral {umbral_efectivo:.3f}",
                "similitud_dominio": sim,
                "metodo":            "semantico_imagebind"
            }

        # Paso 2: LLM como árbitro
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

            if not valido and imagen_activa:
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


# =============================================================================
# EXTRACTOR DE IMÁGENES DE PDF
# =============================================================================

class ExtractorImagenesPDF:
    MIN_WIDTH  = 200
    MIN_HEIGHT = 200

    def __init__(self, directorio_salida: str = DIRECTORIO_IMAGENES):
        self.directorio_salida = directorio_salida
        os.makedirs(directorio_salida, exist_ok=True)

    @staticmethod
    def extraer_caption_imagen(page_fitz, img_bbox, texto_pagina_completo: str) -> str:
        """Extrae el texto más cercano espacialmente a la imagen (caption).
        Busca primero debajo (caption típico), luego arriba.
        También busca referencias 'Fig. X' / 'Imagen X' en el texto."""
        caption = ""
        try:
            # Área de búsqueda: debajo de la imagen (hasta 80 puntos)
            area_abajo = fitz.Rect(
                img_bbox[0], img_bbox[3],
                img_bbox[2], img_bbox[3] + 80
            )
            caption = page_fitz.get_text("text", clip=area_abajo).strip()
            if not caption:
                # Buscar arriba de la imagen (hasta 60 puntos)
                area_arriba = fitz.Rect(
                    img_bbox[0], max(0, img_bbox[1] - 60),
                    img_bbox[2], img_bbox[1]
                )
                caption = page_fitz.get_text("text", clip=area_arriba).strip()
        except Exception:
            pass
        # Buscar referencias "Fig. X" / "Imagen X" en el texto
        referencias = re.findall(
            r'(?:Fig(?:ura)?\.?\s*\d+[A-Za-z]?[^.]*\.)|(?:Imagen\s+\d+[\.\d]*[^.]*\.)',
            texto_pagina_completo,
            re.IGNORECASE
        )
        partes = []
        if caption:
            partes.append(caption)
        if referencias:
            partes.extend(referencias[:3])
        if partes:
            return "\n".join(partes)
        # Fallback: primeros 500 chars del texto de la página
        return texto_pagina_completo[:500] if texto_pagina_completo else ""

    def extraer_de_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        imagenes_extraidas = []
        nombre_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error abriendo {pdf_path}: {e}")
            return []

        def _texto_pagina_con_contexto(doc, idx_0based: int) -> str:
            """Devuelve texto SOLO de la página actual.
            El caption se extrae por proximidad espacial (bbox),
            no por concatenación de páginas adyacentes."""
            try:
                return doc[idx_0based].get_text().strip()
            except Exception:
                return ""

        for num_pagina, pagina in enumerate(doc, start=1):
            idx_0 = num_pagina - 1  # índice base-0 para acceder a doc[idx]
            valid_images_this_page = []
            
            # Método preciso: solo imágenes realmente dibujadas en esta página
            try:
                img_info_list = pagina.get_image_info(xrefs=True)
                # Mapear xref -> bbox para extracción de caption (Fix #1)
                xref_to_bbox = {info["xref"]: info.get("bbox") for info in img_info_list if info.get("xref")}
                page_xrefs = list(xref_to_bbox.keys())
                
                if page_xrefs:
                    for xref in page_xrefs:
                        try:
                            base_image = doc.extract_image(xref)
                            if not base_image: continue
                            
                            image_bytes = base_image["image"]
                            from io import BytesIO
                            pil_temp = Image.open(BytesIO(image_bytes))
                            w, h = pil_temp.size
                            
                            if w >= self.MIN_WIDTH and h >= self.MIN_HEIGHT:
                                valid_images_this_page.append({
                                    "pil": pil_temp,
                                    "area": w * h,
                                    "bbox": xref_to_bbox.get(xref)
                                })
                        except Exception:
                            continue
            except Exception:
                pass
            
            if valid_images_this_page:
                # Seleccionar la imagen más grande (histológica)
                mejor = max(valid_images_this_page, key=lambda x: x["area"])
                pil_img = mejor["pil"]
                img_bbox = mejor.get("bbox")
                nombre_archivo = f"{nombre_pdf}_pag{num_pagina}.png"
                ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
                
                try:
                    pil_img.save(ruta_completa, format="PNG")
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    
                    texto_completo_pagina = _texto_pagina_con_contexto(doc, idx_0)
                    # Extraer caption por proximidad espacial (Fix #1)
                    caption = ""
                    if img_bbox:
                        caption = self.extraer_caption_imagen(pagina, img_bbox, texto_completo_pagina)

                    imagenes_extraidas.append({
                        "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                        "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                        "texto_pagina": texto_completo_pagina,
                        "caption": caption
                    })
                except Exception as e:
                    print(f"  ⚠️ Error guardando pág {num_pagina}: {e}")
            else:
                # FALLBACK: No hay imágenes o son muy chicas -> Renderizar página completa
                try:
                    from pdf2image import convert_from_path
                    pag_imgs = convert_from_path(pdf_path, first_page=num_pagina, last_page=num_pagina, dpi=150)
                    if pag_imgs:
                        pil_full = pag_imgs[0]
                        nombre_archivo = f"{nombre_pdf}_pag{num_pagina}_full.png"
                        ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
                        pil_full.save(ruta_completa, format="PNG")
                        
                        try:
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""

                        texto_completo_pagina = _texto_pagina_con_contexto(doc, idx_0)
                        # Fallback: sin bbox, buscar solo refs "Fig./Imagen"
                        refs = re.findall(
                            r'(?:Fig(?:ura)?\.?\s*\d+[A-Za-z]?[^.]*\.)|(?:Imagen\s+\d+[\.\d]*[^.]*\.)',
                            texto_completo_pagina, re.IGNORECASE
                        )
                        caption_fb = "\n".join(refs[:3]) if refs else ""

                        imagenes_extraidas.append({
                            "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                            "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                            "texto_pagina": texto_completo_pagina,
                            "caption": caption_fb
                        })
                except Exception as e:
                    print(f"  ⚠️ Fallback error pág {num_pagina}: {e}")

        doc.close()
        print(f"  📸 {len(imagenes_extraidas)} imágenes procesadas de {os.path.basename(pdf_path)}")
        return imagenes_extraidas

    def extraer_de_directorio(self, directorio: str) -> List[Dict[str, str]]:
        todas = []
        pdfs  = glob.glob(os.path.join(directorio, "*.pdf"))
        print(f"📂 Extrayendo {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            todas.extend(self.extraer_de_pdf(pdf_path))
        print(f"✅ Total imágenes: {len(todas)}")
        return todas


# =============================================================================
# EXTRACTOR DE TEMARIO
# =============================================================================

class ExtractorTemario:
    def __init__(self, llm):
        self.llm   = llm
        self.temas: List[str] = []

    async def extraer_temario(self, texto_completo: str) -> List[str]:
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
        return "\n".join(f"- {t}" for t in self.temas[:100]) if self.temas else "No disponible."


# =============================================================================
# EXTRACTOR DE ENTIDADES HISTOLÓGICAS
# =============================================================================

class ExtractorEntidades:
    def __init__(self, llm):
        self.llm = llm

    async def extraer_de_texto(self, texto: str) -> Dict[str, List[str]]:
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


# =============================================================================
# ESTADO DEL GRAFO LANGGRAPH
# =============================================================================

class AgentState(TypedDict):
    messages:                    Annotated[list, operator.add]
    consulta_texto:              str
    imagen_path:                 Optional[str]
    imagen_embedding_uni:        Optional[List[float]]
    imagen_embedding_plip:       Optional[List[float]]
    texto_embedding:             Optional[List[float]]
    contexto_memoria:            str
    contenido_base:              str
    terminos_busqueda:           str
    entidades_consulta:          Dict[str, List[str]]
    consulta_busqueda_texto:     str
    consulta_busqueda_visual:    str
    resultados_busqueda:         List[Dict[str, Any]]
    resultados_validos:          List[Dict[str, Any]]
    contexto_documentos:         str
    respuesta_final:             str
    trayectoria:                 List[Dict[str, Any]]
    user_id:                     str
    tiempo_inicio:               float
    analisis_visual:             Optional[str]
    tiene_imagen:                bool
    imagen_es_nueva:             bool           # True si se subió en ESTE turno
    contexto_suficiente:         bool
    temario:                     List[str]
    tema_valido:                 bool
    tema_encontrado:             Optional[str]
    imagenes_recuperadas:        List[str]
    imagenes_texto_map:          Dict[str, str]   # path -> texto descriptivo del manual
    analisis_comparativo:        Optional[str]
    estructura_identificada:     Optional[str]
    similitud_semantica_dominio: float
    confianza_baja:              bool


# =============================================================================
# ASISTENTE PRINCIPAL v4.1
# =============================================================================

class AsistenteHistologiaNeo4j:

    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

    def __init__(self):
        self._setup_apis()
        self.llm             = None
        self.memoria         = None
        self.graph           = None
        self.compiled_graph  = None
        self.memory_saver    = None
        self.contenido_base  = ""

        self.uni   = None
        self.plip  = None
        self.embeddings = None
        self.embed_dim = DIM_TEXTO

        self.neo4j: Optional[Neo4jClient] = None

        self.extractor_imagenes       = ExtractorImagenesPDF(DIRECTORIO_IMAGENES)
        self.extractor_temario        = None
        self.extractor_entidades      = None
        self.clasificador_semantico: Optional[ClasificadorSemantico] = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"⚠️ GPU incompatible detectada (sm_{cap[0]}{cap[1]}). Forzando CPU para evitar fallback_error.")
                    self.device = "cpu"
            except:
                pass
        print(f"✅ AsistenteHistologiaNeo4j v4.2 inicializado en {self.device}")

    def _setup_apis(self):
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY") or ""
        print("✅ APIs configuradas")

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    async def inicializar_componentes(self):
        self._init_modelos()
        self.memoria             = SemanticMemory(
            llm=self.llm,
            embeddings=self.embeddings,
            uni=self.uni,
            plip=self.plip
        )
        self.extractor_temario   = ExtractorTemario(llm=self.llm)
        self.extractor_entidades = ExtractorEntidades(llm=self.llm)
        self.clasificador_semantico = ClasificadorSemantico(
            llm=self.llm,
            embeddings=self.embeddings, # Gemini
            device=self.device,
            temario=[]   # se actualizará tras extraer el temario
        )

        self.neo4j = Neo4jClient(
            uri      = userdata.get("NEO4J_URI")      or os.getenv("NEO4J_URI"),
            user     = userdata.get("NEO4J_USERNAME")  or os.getenv("NEO4J_USERNAME", "neo4j"),
            password = userdata.get("NEO4J_PASSWORD")  or os.getenv("NEO4J_PASSWORD"),
        )
        await self.neo4j.connect()
        await self.neo4j.crear_esquema()

        self.memory_saver   = MemorySaver()
        self._crear_grafo()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory_saver)
        print("✅ Todos los componentes inicializados")

    def _init_modelos(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=userdata.get("GROQ_API_KEY"),
            temperature=0, max_retries=1
        )
        print("✅ Groq inicializado")
        
        # Inicializar Embeddings (HuggingFace)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        print("✅ Embeddings HuggingFace inicializados")

        # Cargar Modelos (UNI + PLIP)
        self.plip = PlipWrapper(self.device)
        self.plip.load()
        
        self.uni = UniWrapper(self.device)
        self.uni.load()

    def _init_imagebind(self):
        pass # ImageBind eliminado

    # ------------------------------------------------------------------
    # Grafo LangGraph v4.1
    # ------------------------------------------------------------------

    def _crear_grafo(self):
        g = StateGraph(AgentState)

        g.add_node("inicializar",          self._nodo_inicializar)
        g.add_node("procesar_imagen",      self._nodo_procesar_imagen)
        g.add_node("clasificar",           self._nodo_clasificar)
        g.add_node("generar_consulta",     self._nodo_generar_consulta)
        g.add_node("buscar_neo4j",         self._nodo_buscar_neo4j)
        g.add_node("filtrar_contexto",     self._nodo_filtrar_contexto)
        g.add_node("analisis_comparativo", self._nodo_analisis_comparativo)
        g.add_node("generar_respuesta",    self._nodo_generar_respuesta)
        g.add_node("finalizar",            self._nodo_finalizar)
        g.add_node("fuera_temario",        self._nodo_fuera_temario)

        g.add_edge(START,                  "inicializar")

        # Router: si hay imagen → procesar_imagen, si solo texto → clasificar
        g.add_conditional_edges(
            "inicializar",
            self._route_por_modo,
            {"con_imagen": "procesar_imagen", "solo_texto": "clasificar"}
        )
        g.add_edge("procesar_imagen",      "clasificar")

        g.add_conditional_edges(
            "clasificar",
            self._route_por_temario,
            {"en_temario": "generar_consulta", "fuera_temario": "fuera_temario"}
        )
        g.add_edge("fuera_temario",        "finalizar")
        g.add_edge("generar_consulta",     "buscar_neo4j")
        g.add_edge("buscar_neo4j",         "filtrar_contexto")

        # Router: si hay imagen → analisis_comparativo, si no → generar_respuesta
        g.add_conditional_edges(
            "filtrar_contexto",
            self._route_analisis_comparativo,
            {"con_imagen": "analisis_comparativo", "sin_imagen": "generar_respuesta"}
        )
        g.add_edge("analisis_comparativo", "generar_respuesta")
        g.add_edge("generar_respuesta",    "finalizar")
        g.add_edge("finalizar",            END)

        self.graph = g

    def _route_por_modo(self, state: AgentState) -> str:
        """Decide si procesar imagen o ir directo a clasificar (texto puro)."""
        imagen_path = state.get("imagen_path")
        tiene_imagen_nueva = imagen_path and os.path.exists(imagen_path)
        tiene_imagen_memoria = self.memoria and self.memoria.tiene_imagen_previa()
        
        # Detectar si la consulta hace referencia explícita a una imagen
        consulta_texto = state.get("consulta_texto", "").lower()
        palabras_referencia_imagen = [
            "imagen", "foto", "esta", "esto", "este", "esa", "eso", "ese",
            "la imagen", "la foto", "qué es esto", "qué es esta", "qué tipo",
            "identifica", "analiza", "describe", "observa", "muestra",
            "se ve", "veo", "aparece", "presenta"
        ]
        hace_referencia_imagen = any(palabra in consulta_texto for palabra in palabras_referencia_imagen)
        
        # Lógica de routing:
        # 1. Si hay imagen nueva → siempre modo con_imagen
        # 2. Si hay imagen en memoria Y la consulta hace referencia → modo con_imagen
        # 3. Si hay imagen en memoria pero la consulta NO hace referencia → modo solo_texto
        # 4. Si no hay imagen → modo solo_texto
        
        if tiene_imagen_nueva:
            modo = "con_imagen"
            print("🖼️ Modo multimodal detectado (imagen nueva)")
        elif tiene_imagen_memoria and hace_referencia_imagen:
            modo = "con_imagen"
            print("🖼️ Modo multimodal detectado (imagen en memoria + referencia en consulta)")
        else:
            modo = "solo_texto"
            if tiene_imagen_memoria:
                print("📝 Modo solo texto — consulta teórica (ignorando imagen en memoria)")
            else:
                print("📝 Modo solo texto — sin imagen disponible")
        
        # Registrar modo en trayectoria
        state["trayectoria"].append({
            "nodo": "Router:_route_por_modo",
            "modo_detectado": modo,
            "tiene_imagen_nueva": tiene_imagen_nueva,
            "tiene_imagen_memoria": tiene_imagen_memoria,
            "hace_referencia_imagen": hace_referencia_imagen
        })
        
        return modo

    def _route_analisis_comparativo(self, state: AgentState) -> str:
        """Salta análisis comparativo si no hay imagen."""
        if state.get("tiene_imagen") and state.get("imagen_path"):
            return "con_imagen"
        return "sin_imagen"

    def _route_por_temario(self, state: AgentState) -> str:
        return "en_temario"

    # ------------------------------------------------------------------
    # Nodos
    # ------------------------------------------------------------------

    async def _nodo_inicializar(self, state: AgentState) -> AgentState:
        print("📝 Inicializando flujo v4.1 (Neo4j)")
        state["contexto_memoria"]            = self.memoria.get_context(state.get("consulta_texto", ""))
        state["contenido_base"]              = self.contenido_base
        state["tiempo_inicio"]               = time.time()
        state["tiene_imagen"]                = False
        state["imagen_es_nueva"]             = False
        state["contexto_suficiente"]         = False
        state["resultados_validos"]          = []
        state["terminos_busqueda"]           = ""
        state["entidades_consulta"]          = {"tejidos": [], "estructuras": [], "tinciones": []}
        state["imagenes_recuperadas"]        = []
        state["tema_valido"]                 = True
        state["tema_encontrado"]             = None
        state["temario"]                     = self.extractor_temario.temas if self.extractor_temario else []
        state["analisis_comparativo"]        = None
        state["estructura_identificada"]     = None
        state["texto_embedding"]             = None
        state["similitud_semantica_dominio"] = 0.0
        state["trayectoria"] = [{"nodo": "Inicializar", "tiempo": 0}]
        return state

    async def _nodo_procesar_imagen(self, state: AgentState) -> AgentState:
        """
        Tres casos:
          1. Imagen nueva este turno → procesarla y registrarla en memoria.
          2. Sin imagen nueva pero hay activa en memoria → reutilizarla.
          3. Sin imagen en ningún lado → modo texto puro.
        """
        t0 = time.time()
        print("🖼️ Procesando imagen...")

        imagen_path_nuevo = state.get("imagen_path")
        imagen_es_nueva   = False

        if imagen_path_nuevo and os.path.exists(imagen_path_nuevo):
            imagen_path_activo = imagen_path_nuevo
            imagen_es_nueva    = True
            self.memoria.set_imagen(imagen_path_activo)
            print(f"   🆕 Nueva imagen: {imagen_path_activo}")

        elif self.memoria.tiene_imagen_previa():
            imagen_path_activo = self.memoria.get_imagen_activa()
            state["imagen_path"] = imagen_path_activo
            print(f"   ♻️  Reutilizando imagen del turno "
                  f"{self.memoria.imagen_turno_subida}: "
                  f"{os.path.basename(imagen_path_activo)}")

        else:
            imagen_path_activo = None

        if imagen_path_activo and os.path.exists(imagen_path_activo):
            try:
                # emb_c removed
                emb_u = self.uni.embed_image(imagen_path_activo)
                emb_p = self.plip.embed_image(imagen_path_activo)
                
                state["imagen_embedding_uni"]   = emb_u.tolist()
                state["imagen_embedding_plip"]  = emb_p.tolist()
                
                state["tiene_imagen"]     = True
                state["imagen_es_nueva"]  = imagen_es_nueva

                if imagen_es_nueva or not state.get("analisis_visual"):
                    state["analisis_visual"] = await self._describir_imagen_histologica(
                        imagen_path_activo
                    )
                    print(f"   🔬 Análisis visual generado ({len(state['analisis_visual'])} chars)")
                else:
                    print("   ♻️  Reutilizando análisis visual previo del contexto")

                print(f"✅ Imagen lista | nueva={imagen_es_nueva}")
            except Exception as e:
                print(f"❌ Error imagen: {e}")
                import traceback; traceback.print_exc()
                state["imagen_embedding_uni"]   = None
                state["imagen_embedding_plip"]  = None
                state["analisis_visual"]  = None
                state["tiene_imagen"]     = False
        else:
            print("ℹ️ Sin imagen — modo texto")
            state["imagen_embedding"] = None
            state["analisis_visual"]  = None
            state["tiene_imagen"]     = False
            state["imagen_es_nueva"]  = False

        state["trayectoria"].append({
            "nodo":            "ProcesarImagen",
            "tiene_imagen":    state["tiene_imagen"],
            "imagen_es_nueva": imagen_es_nueva,
            "tiempo":          round(time.time()-t0, 2)
        })
        return state

    async def _describir_imagen_histologica(self, imagen_path: str) -> str:
        try:
            with open(imagen_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext  = os.path.splitext(imagen_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            features_lista = "\n".join(
                f"  {i+1}. {f}" for i, f in enumerate(FEATURES_DISCRIMINATORIAS)
            )
            msg = HumanMessage(content=[
                {"type": "text", "text": (
                    "Describe esta imagen histológica.\n\n"
                    "PARTE 1 — DESCRIPCIÓN GENERAL: tipo tejido, coloración, aumento, estructuras.\n\n"
                    f"PARTE 2 — FEATURES DISCRIMINATORIAS:\n{features_lista}\n\n"
                    "PARTE 3 — DIAGNÓSTICO DIFERENCIAL: 3 estructuras más probables, "
                    "diferencias morfológicas, ¿confundible con cuerpo de albicans?"
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
            ])
            resp = await invoke_con_reintento(self.llm, [msg])
            return resp.content
        except Exception as e:
            print(f"⚠️ Error describiendo imagen: {e}")
            return ""

    async def _nodo_clasificar(self, state: AgentState) -> AgentState:
        """
        1. Extrae términos histológicos y entidades para la búsqueda.
        2. Usa ClasificadorSemantico (ImageBind + LLM) para verificar dominio.
           Reemplaza la verificación por keywords de v4.0.
        """
        t0 = time.time()
        print("🔍 Clasificando consulta (semántico v4.1)...")

        # ── Extracción de términos ─────────────────────────────────────
        system = (
            "Extrae términos técnicos histológicos de la consulta.\n"
            "Devuelve:\nTEJIDO: [...]\nESTRUCTURA: [...]\nCONCEPTO: [...]\n"
            "TINCIÓN: [...]\nTÉRMINOS_CLAVE: [...]"
        )
        partes = [f"CONSULTA:\n{state['consulta_texto']}"]
        analisis_visual = _safe(state.get("analisis_visual"))
        if analisis_visual:
            partes.append(f"ANÁLISIS VISUAL:\n{analisis_visual[:600]}")
        contexto_mem = _safe(state.get("contexto_memoria"))
        if contexto_mem and contexto_mem != "No hay consultas previas.":
            partes.append(f"CONTEXTO:\n{contexto_mem[:300]}")

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content="\n\n".join(partes))
            ])
            state["terminos_busqueda"] = resp.content
        except Exception as e:
            state["terminos_busqueda"] = state["consulta_texto"]

        # ── Entidades para búsqueda por grafo ──────────────────────────
        texto_para_entidades = (
            state["consulta_texto"] + " " + _safe(state.get("analisis_visual"))
        )
        state["entidades_consulta"] = await self.extractor_entidades.extraer_de_texto(
            texto_para_entidades
        )
        print(f"   🏷️ Entidades: {state['entidades_consulta']}")

        # ── Embedding del texto de consulta (Gemini) ───────────────────
        try:
            emb_texto = self._embed_texto(state["consulta_texto"])
            state["texto_embedding"] = emb_texto
        except Exception as e:
            print(f"⚠️ Error embedding texto: {e}")
            state["texto_embedding"] = None

        # ── Clasificación semántica de dominio ─────────────────────────
        verificacion = await self.clasificador_semantico.clasificar(
            consulta       = state["consulta_texto"],
            analisis_visual= state.get("analisis_visual"),
            imagen_activa  = state.get("tiene_imagen", False),
            temario_muestra= state.get("temario", [])[:60],
        )

        state["tema_valido"]                 = verificacion.get("valido", True)
        state["tema_encontrado"]             = verificacion.get("tema_encontrado")
        state["similitud_semantica_dominio"] = verificacion.get("similitud_dominio", 0.0)

        print(f"   📚 Válido: {state['tema_valido']} | "
              f"Tema: {state['tema_encontrado'] or 'N/A'} | "
              f"Sim: {state['similitud_semantica_dominio']:.3f} | "
              f"Método: {verificacion.get('metodo')}")

        state["trayectoria"].append({
            "nodo":                  "Clasificar",
            "tema_valido":           state["tema_valido"],
            "tema_encontrado":       state["tema_encontrado"],
            "entidades":             state["entidades_consulta"],
            "similitud_dominio":     state["similitud_semantica_dominio"],
            "metodo_clasificacion":  verificacion.get("metodo"),
            "tiempo":                round(time.time()-t0, 2)
        })
        return state

    async def _nodo_fuera_temario(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("🚫 Consulta fuera del dominio histológico")
        temario = state.get("temario") or []
        muestra = "\n".join(f"  • {t}" for t in temario[:20])
        if len(temario) > 20:
            muestra += f"\n  ... y {len(temario)-20} más"
        state["respuesta_final"] = (
            "⚠️ **Consulta fuera del dominio disponible**\n\n"
            "Tu consulta no parece estar relacionada con histología, patología "
            "o morfología tisular/celular.\n\n"
            f"**Temas disponibles (muestra):**\n{muestra}\n\n"
            "Si tenés una imagen histológica, subila y reformulá tu pregunta. "
            "Ejemplos válidos: '¿qué tipo de tejido es este?', "
            "'describe la estructura observada', 'diagnóstico diferencial'."
        )
        state["contexto_suficiente"] = False
        state["trayectoria"].append({"nodo": "FueraTemario", "tiempo": round(time.time()-t0, 2)})
        return state

    async def _nodo_generar_consulta(self, state: AgentState) -> AgentState:
        t0 = time.time()
        tema_extra = f"\nTEMA: {state['tema_encontrado']}" if state.get("tema_encontrado") else ""
        system = (
            "Genera consultas cortas (≤8 palabras) para histología.\n"
            "Formato:\nCONSULTA_TEXTO: <texto>\n"
            + ("CONSULTA_VISUAL: <visual>" if state.get("tiene_imagen") else "")
        )
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system),
                HumanMessage(content=(
                    f"TÉRMINOS:\n{_safe(state.get('terminos_busqueda'))}"
                    f"{tema_extra}\nCONSULTA: {state['consulta_texto']}"
                ))
            ])
            contenido = resp.content
            ct = state["consulta_texto"][:77]
            cv = ""
            if "CONSULTA_TEXTO:" in contenido:
                after = contenido.split("CONSULTA_TEXTO:")[1]
                if "CONSULTA_VISUAL:" in after:
                    ct = after.split("CONSULTA_VISUAL:")[0].strip()[:77]
                    cv = after.split("CONSULTA_VISUAL:")[1].strip()[:77]
                else:
                    ct = after.strip()[:77]
            state["consulta_busqueda_texto"]  = ct
            state["consulta_busqueda_visual"] = cv
        except Exception as e:
            state["consulta_busqueda_texto"]  = state["consulta_texto"][:77]
            state["consulta_busqueda_visual"] = ""

        print(f"   📝 query='{state['consulta_busqueda_texto']}'")
        state["trayectoria"].append({
            "nodo": "GenerarConsulta", "query": state["consulta_busqueda_texto"],
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_buscar_neo4j(self, state: AgentState) -> AgentState:
        t0 = time.time()
        print("📚 Búsqueda híbrida Neo4j...")

        resultados = await self.neo4j.busqueda_hibrida(
            texto_embedding        = state.get("texto_embedding"),
            imagen_embedding_uni   = state.get("imagen_embedding_uni"),
            imagen_embedding_plip  = state.get("imagen_embedding_plip"),
            entidades              = state.get("entidades_consulta", {}),
            top_k                  = 10
        )

        tejidos = state.get("entidades_consulta", {}).get("tejidos", [])
        if len(tejidos) >= 2:
            camino = await self.neo4j.busqueda_camino_semantico(tejidos[0], tejidos[1])
            if camino:
                print(f"   🗺️ Camino semántico: {len(camino)} nodos")
                resultados.extend(camino)

        state["resultados_busqueda"] = resultados
        print(f"✅ {len(resultados)} resultados")

        state["trayectoria"].append({
            "nodo": "BuscarNeo4j", "hits": len(resultados),
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_filtrar_contexto(self, state: AgentState) -> AgentState:
        t0     = time.time()
        umbral_imagen = self.SIMILARITY_THRESHOLD
        es_solo_texto = not state.get("tiene_imagen", False)

        # Umbral de texto más permisivo en modo solo texto
        umbral_texto = 0.45 if es_solo_texto else 0.6

        validos = []
        for r in state["resultados_busqueda"]:
            current_sim = r.get("similitud", 0)
            if r.get("tipo") == "texto" and current_sim < umbral_texto:
                continue
            if r.get("tipo") == "imagen" and current_sim < umbral_imagen:
                continue

            # Si es imagen pero no existe el archivo en disco, lo rechazamos
            if r.get("tipo") == "imagen":
                img_p = r.get("imagen_path")
                if not img_p or not os.path.exists(img_p):
                    continue
            validos.append(r)

        state["resultados_validos"]  = validos
        state["contexto_suficiente"] = len(validos) > 0

        vistas: set = set()
        imagenes_unicas: List[str] = []
        imagenes_texto: Dict[str, str] = {}
        for r in validos:
            img_path = r.get("imagen_path")
            if img_path and os.path.exists(img_path) and img_path not in vistas:
                vistas.add(img_path)
                imagenes_unicas.append(img_path)
                imagenes_texto[img_path] = _safe(r.get('texto', ''))[:500]
        state["imagenes_recuperadas"] = imagenes_unicas
        state["imagenes_texto_map"]   = imagenes_texto

        if validos:
            validos_sorted = sorted(validos, key=lambda x: x.get("similitud", 0), reverse=True)
            bloques = []
            for i, r in enumerate(validos_sorted, 1):
                enc = (f"[Sección {i} | Fuente: {r.get('fuente','N/A')} | "
                       f"Tipo: {r.get('tipo','?')} | Sim: {r.get('similitud',0):.3f}")
                if r.get("imagen_path"):
                    enc += f" | Imagen: {os.path.basename(r['imagen_path'])}"
                enc += "]"
                bloques.append(f"{enc}\n{_safe(r.get('texto',''))[:700]}")
            state["contexto_documentos"] = "\n\n".join(bloques)
            modo_str = "TEXTO" if es_solo_texto else "IMAGEN+TEXTO"
            print(f"✅ {len(validos)} válidos | {len(imagenes_unicas)} imgs | Modo: {modo_str}")
        else:
            state["contexto_documentos"] = ""
            print(f"⚠️ Ningún resultado supera umbral (texto={umbral_texto}, img={umbral_imagen})")

        state["trayectoria"].append({
            "nodo": "FiltrarContexto", "hits_validos": len(validos),
            "imgs": len(state["imagenes_recuperadas"]),
            "modo": "solo_texto" if es_solo_texto else "multimodal",
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_analisis_comparativo(self, state: AgentState) -> AgentState:
        t0 = time.time()

        if not state.get("tiene_imagen") or not state.get("imagen_path"):
            print("ℹ️ Sin imagen — análisis comparativo omitido")
            state["trayectoria"].append({
                "nodo": "AnalisisComparativo", "motivo": "sin imagen",
                "tiempo": round(time.time()-t0, 2)
            })
            return state

        imagenes_ref = [
            p for p in state.get("imagenes_recuperadas", [])[:3] if os.path.exists(p)
        ]
        if not imagenes_ref:
            print("ℹ️ Sin referencias — análisis comparativo omitido")
            state["analisis_comparativo"] = None
            state["trayectoria"].append({
                "nodo": "AnalisisComparativo", "motivo": "sin referencias",
                "tiempo": round(time.time()-t0, 2)
            })
            return state

        print(f"🔬 Análisis comparativo vs {len(imagenes_ref)} referencias...")
        content_parts = [{"type": "text", "text": (
            "Compara la imagen de consulta con las referencias del manual para "
            "determinar si corresponden a la misma estructura histológica.\n\n"
            "=== IMAGEN DE CONSULTA ==="
        )}]

        try:
            with open(state["imagen_path"], "rb") as f:
                data_u = base64.b64encode(f.read()).decode("utf-8")
            ext  = os.path.splitext(state["imagen_path"])[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data_u}"}
            })
        except Exception as e:
            print(f"⚠️ No se pudo cargar imagen usuario: {e}")
            state["analisis_comparativo"] = None
            return state

        analisis_previo = _safe(state.get("analisis_visual"))
        if analisis_previo:
            content_parts.append({"type": "text", "text": (
                f"\nAnálisis previo:\n{analisis_previo[:600]}\n"
            )})

        imagenes_texto = state.get("imagenes_texto_map", {})
        for i, ref_path in enumerate(imagenes_ref, 1):
            texto_ref = imagenes_texto.get(ref_path, "Sin descripción disponible")
            content_parts.append({"type": "text", "text": (
                f"\n=== REFERENCIA #{i} ({os.path.basename(ref_path)}) ===\n"
                f"DESCRIPCIÓN DEL MANUAL: {texto_ref}"
            )})
            try:
                with open(ref_path, "rb") as f:
                    data_r = base64.b64encode(f.read()).decode("utf-8")
                ext  = os.path.splitext(ref_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data_r}"}
                })
            except Exception as e:
                print(f"  ⚠️ No se pudo cargar {ref_path}: {e}")

        features_lista = "\n".join(f"  - {f}" for f in FEATURES_DISCRIMINATORIAS)
        content_parts.append({"type": "text", "text": (
            "\n=== INSTRUCCIONES ESTRICTAS ===\n"
            f"Compara rigurosamente basándote en:\n{features_lista}\n\n"
            "IMPORTANTE: Cada referencia incluye la DESCRIPCIÓN DEL MANUAL. "
            "Usa esa descripción como fuente de verdad para identificar la estructura, "
            "NO tu propia interpretación visual.\n\n"
            "TU ROL ES SER UN EVALUADOR OBJETIVO. Tu objetivo principal es determinar verdaderamente si la imagen de consulta y las referencias muestran la misma estructura histológica.\n\n"
            "1. TABLA COMPARATIVA (Markdown): | Feature | Consulta | Ref#1 | Ref#2 |\n"
            "2. VEREDICTO DE IDENTIDAD: Evalúa las similitudes morfológicas y declara si son el MISMO TEJIDO o TEJIDOS DIFERENTES.\n"
            "3. CONCLUSIÓN FINAL: Debes terminar con una de estas dos frases EXACTAS:\n"
            "   - Si coinciden: 'CONCLUSIÓN: SÍ son la misma estructura histológica'\n"
            "   - Si NO coinciden: 'CONCLUSIÓN: TEJIDOS DIFERENTES'\n"
            "   Si hay dudas pero las similitudes son significativas, considera que SÍ coinciden."
        )})

        try:
            resp = await invoke_con_reintento(self.llm, [HumanMessage(content=content_parts)])
            state["analisis_comparativo"] = resp.content
            state["estructura_identificada"] = await self._extraer_estructura(resp.content)
            print(f"✅ Análisis comparativo: {len(resp.content)} chars")
            print(f"   → Estructura: {state['estructura_identificada']}")
        except Exception as e:
            print(f"❌ Error análisis comparativo: {e}")
            state["analisis_comparativo"] = None
            state["estructura_identificada"] = None

        state["trayectoria"].append({
            "nodo": "AnalisisComparativo", "refs": len(imagenes_ref),
            "estructura": state.get("estructura_identificada"),
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _extraer_estructura(self, analisis: str) -> Optional[str]:
        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=(
                    "Extrae el nombre de la estructura histológica más probable. Solo el nombre."
                )),
                HumanMessage(content=analisis[-1000:])
            ])
            return resp.content.strip()
        except Exception:
            return None

    async def _nodo_generar_respuesta(self, state: AgentState) -> AgentState:
        t0 = time.time()
        es_solo_texto = not state.get("tiene_imagen", False)
        modo_str = "TEXTO" if es_solo_texto else "MULTIMODAL"
        print(f"💭 Generando respuesta v4.2 [{modo_str}]...")

        if not state["contexto_suficiente"]:
            if es_solo_texto:
                # Modo texto sin contexto: respuesta más amable
                print("   ⚠️ Sin contexto RAG para consulta de texto")
                temario = state.get("temario") or []
                muestra = "\n".join(f"  • {t}" for t in temario[:15])
                if len(temario) > 15:
                    muestra += f"\n  ... y {len(temario)-15} más"
                state["respuesta_final"] = (
                    "⚠️ **No encontré información específica sobre eso en el manual**\n\n"
                    "La consulta es válida pero no encontré contenido suficiente en la "
                    "base de datos del manual para responderla con precisión.\n\n"
                    f"**Temas disponibles en el manual (muestra):**\n{muestra}\n\n"
                    "Podés intentar:\n"
                    "- Reformular la pregunta con términos más específicos\n"
                    "- Preguntar sobre alguno de los temas listados arriba\n"
                    "- Subir una imagen histológica para análisis visual"
                )
            else:
                # Modo imagen sin contexto: rechazo estricto
                print("   ⚠️ Sin contexto RAG — rechazo (imagen no encontrada en DB)")
                temario = state.get("temario") or []
                muestra = "\n".join(f"  • {t}" for t in temario[:20])
                if len(temario) > 20:
                    muestra += f"\n  ... y {len(temario)-20} más"
                state["respuesta_final"] = (
                    "⚠️ **Consulta fuera del dominio disponible**\n\n"
                    "Tu consulta no parece estar relacionada con histología, patología "
                    "o morfología tisular/celular.\n\n"
                    f"**Temas disponibles (muestra):**\n{muestra}\n\n"
                    "Si tenés una imagen histológica, subila y reformulá tu pregunta. "
                    "Ejemplos válidos: '¿qué tipo de tejido es este?', "
                    "'describe la estructura observada', 'diagnóstico diferencial'."
                )
            state["trayectoria"].append({
                "nodo": "GenerarRespuesta", "contexto_suficiente": False,
                "modo": "solo_texto" if es_solo_texto else "multimodal",
                "tiempo": round(time.time()-t0, 2)
            })
            return state

        tiene_comparativo = bool(_safe(state.get("analisis_comparativo")))

        # ── System prompt diferenciado según modo ──────────────────────
        if es_solo_texto:
            system_prompt = (
                "Eres un asistente experto de histología. Respondés consultas de texto "
                "basándote EXCLUSIVAMENTE en el contenido del manual/base de datos.\n\n"
                "REGLAS FUNDAMENTALES:\n"
                "1. FUENTE DE VERDAD: Usá SOLO la información de las SECCIONES DEL MANUAL proporcionadas.\n"
                "2. CITAS: Citá las fuentes con [Manual: archivo].\n"
                "3. NO inventes información que no esté en las secciones proporcionadas.\n"
                "4. Si la información es parcial, indicá qué partes provienen del manual y cuáles no están disponibles.\n"
                "5. No diagnósticos clínicos salvo que estén explícitos en el manual.\n\n"
                "ESTRUCTURA DE RESPUESTA:\n"
                "1. Respuesta directa a la consulta basada en el manual\n"
                "2. Características histológicas relevantes según la base de datos\n"
                "3. Fuentes y referencias del manual\n"
                "4. Conclusión"
            )
        else:
            nota_comp = (
                "\n\nIMPORTANTE: El análisis comparativo tiene PRIORIDAD en el diagnóstico diferencial."
                if tiene_comparativo else ""
            )
            regla_validacion = (
                "2. VALIDACIÓN: Revisa el 'ANÁLISIS COMPARATIVO'. "
                "Si la conclusión final del análisis dice explícitamente 'TEJIDOS DIFERENTES' o 'NO son la misma estructura', "
                "entonces debes decir EXACTAMENTE: "
                "'no esta en base de datos: [Manual: No disponible] | [Imagen: NUEVA IMAGEN DEL USUARIO]' "
                "y no escribir nada más. "
                "Si el análisis menciona 'similitudes', 'coinciden', 'mismo tejido', o 'SÍ son la misma estructura', "
                "entonces asume que SÍ coinciden y continúa con el paso 3 detallando el tejido según el manual.\n"
            )
            system_prompt = (
                "Eres un asistente de histología. Responde SOLO con el contenido del manual o la imagen visible en el chat.\n\n"
                "REGLAS FUNDAMENTALES:\n"
                "1. PRIORIDAD ABSOLUTA: La DESCRIPCIÓN TEXTUAL DEL MANUAL es la fuente de verdad. "
                   "Si el texto del manual dice 'Tejido nervioso corteza cerebelosa', ESO es lo correcto, "
                   "sin importar tu propia interpretación visual de la imagen.\n"
                "2. Cita: [Manual: archivo] | [Imagen: archivo]\n"
                "3. Para cada 'IMAGEN DE REFERENCIA', indica el nombre y la descripción textual del manual.\n"
                "4. NO hagas diagnósticos propios basados en tu interpretación visual. "
                   "Usa SIEMPRE el texto del manual asociado a la imagen.\n"
                "5. No diagnósticos clínicos salvo que estén explícitos.\n\n"
                "ESTRUCTURA:\n"
                "1. Análisis de la consulta basado en la imagen del usuario (si la hay)\n"
                f"{regla_validacion}"
                "3. Características histológicas según la base de datos (SOLO lo que dice el texto del manual)\n"
                "4. Conclusión y confianza"
                f"{nota_comp}"
            )

        analisis_comp_str   = _safe(state.get("analisis_comparativo"))
        estructura_str      = _safe(state.get("estructura_identificada"))
        analisis_visual_str = _safe(state.get("analisis_visual"), "No disponible")
        contexto_mem_str    = _safe(state.get("contexto_memoria"))
        terminos_str        = _safe(state.get("terminos_busqueda"))
        tema_str            = _safe(state.get("tema_encontrado"), "N/A")
        entidades_str       = json.dumps(state.get("entidades_consulta", {}), ensure_ascii=False)

        seccion_comp = (f"\n\n**ANÁLISIS COMPARATIVO:**\n{analisis_comp_str[:2000]}"
                        if analisis_comp_str else "")
        seccion_est  = (f"\n\n**ESTRUCTURA IDENTIFICADA:** {estructura_str}"
                        if estructura_str else "")

        content_parts = [{"type": "text", "text": (
            f"**CONSULTA:** {state['consulta_texto']}\n\n"
            f"**HISTORIAL:** {contexto_mem_str[:300]}\n\n"
            f"**TÉRMINOS:** {terminos_str[:300]}\n\n"
            f"**ENTIDADES (grafo):** {entidades_str}\n\n"
            f"**TEMA:** {tema_str}\n\n"
            f"**ANÁLISIS VISUAL USUARIO:**\n{analisis_visual_str[:800]}\n\n"
            f"**SECCIONES DEL MANUAL:**\n{state['contexto_documentos']}"
            f"{seccion_comp}{seccion_est}\n\n"
            "Responde EXCLUSIVAMENTE con el contenido del manual e imágenes de referencia."
        )}]

        imagen_path = state.get("imagen_path")
        if state.get("tiene_imagen") and imagen_path and os.path.exists(imagen_path):
            try:
                with open(imagen_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext  = os.path.splitext(imagen_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                label = ("NUEVA IMAGEN DEL USUARIO" if state.get("imagen_es_nueva")
                         else f"IMAGEN ACTIVA (turno {self.memoria.imagen_turno_subida})")
                content_parts.append({"type": "text", "text": f"\n**{label}:**"})
                content_parts.append({"type": "image_url",
                                       "image_url": {"url": f"data:{mime};base64,{data}"}})
                print(f"   📷 {label}")
            except Exception as e:
                print(f"   ⚠️ No se pudo añadir imagen usuario: {e}")

        imagenes_usadas = 0
        for img_path in state.get("imagenes_recuperadas", [])[:3]:
            if not os.path.exists(img_path):
                continue
            try:
                with open(img_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext    = os.path.splitext(img_path)[1].lower()
                mime   = "image/png" if ext == ".png" else "image/jpeg"
                nombre = os.path.basename(img_path)
                content_parts.append({"type": "text",
                                       "text": f"\n**REFERENCIA [Imagen: {nombre}]:**"})
                content_parts.append({"type": "image_url",
                                       "image_url": {"url": f"data:{mime};base64,{data}"}})
                imagenes_usadas += 1
            except Exception as e:
                print(f"   ⚠️ {img_path}: {e}")

        print(f"   📊 {1 if state.get('tiene_imagen') else 0} usuario + {imagenes_usadas} manual")

        try:
            resp = await invoke_con_reintento(self.llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_parts)
            ])
            if state.get("confianza_baja"):
                warning = "⚠️ *Nota: Esta respuesta se generó con confianza baja (<71% de coincidencia) en los apuntes, tómalo en cuenta.*"
                if "Hola, ¿en qué te puedo ayudar sobre histología?" in resp.content:
                    state["respuesta_final"] = resp.content
                else:
                    state["respuesta_final"] = f"{warning}\n\n{resp.content}"
            else:
                state["respuesta_final"] = resp.content
            print(f"✅ Respuesta: {len(resp.content)} chars")
        except Exception as e:
            print(f"❌ Error: {e}")
            state["respuesta_final"] = f"Error: {e}"

        state["trayectoria"].append({
            "nodo": "GenerarRespuesta", "contexto_suficiente": True,
            "imagenes_usadas": imagenes_usadas, "tiene_comparativo": tiene_comparativo,
            "tiempo": round(time.time()-t0, 2)
        })
        return state

    async def _nodo_finalizar(self, state: AgentState) -> AgentState:
        # Guardar siempre en memoria, no solo cuando hay contexto suficiente
        if state.get("respuesta_final"):
            self.memoria.add_interaction(state["consulta_texto"], state["respuesta_final"])

        total = round(time.time() - state["tiempo_inicio"], 2)
        state["trayectoria"].append({"nodo": "Finalizar", "tiempo_total": total})

        with open("trayectoria_neo4j.json", "w", encoding="utf-8") as f:
            json.dump({
                "trayectoria":             state["trayectoria"],
                "estructura_identificada": state.get("estructura_identificada"),
                "imagenes_recuperadas":    state.get("imagenes_recuperadas", []),
                "entidades_consulta":      state.get("entidades_consulta", {}),
            }, f, indent=4, ensure_ascii=False)

        print(f"✅ Flujo v4.1 completado en {total}s")
        if state.get("estructura_identificada"):
            print(f"   → Estructura: {state['estructura_identificada']}")
        return state

    # ------------------------------------------------------------------
    # Embeddings (Gemini Text)
    # ------------------------------------------------------------------

    def _embed_texto(self, texto: str) -> List[float]:
        # Usa langchain GoogleGenerativeAIEmbeddings
        return embed_query_con_reintento(self.embeddings, texto)


    # ------------------------------------------------------------------
    # Indexación en Neo4j
    # ------------------------------------------------------------------

    def _leer_pdf(self, path: str) -> str:
        try:
            doc = fitz.open(path)
            texto = "".join(page.get_text() for page in doc)
            doc.close()
            return texto
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            return ""

    def _chunks(self, texto: str, size: int = 500) -> List[str]:
        return [texto[i:i+size] for i in range(0, len(texto), size)]

    def procesar_contenido_base(self, directorio: str = DIRECTORIO_PDFS) -> str:
        pdfs = glob.glob(os.path.join(directorio, "*.pdf"))
        if not pdfs:
            print(f"⚠️ Sin PDFs en {directorio}")
            return ""
        self.contenido_base = "\n".join(self._leer_pdf(p) for p in pdfs)
        print(f"📚 {len(pdfs)} PDFs leídos ({len(self.contenido_base)} chars)")
        return self.contenido_base[:500]

    async def extraer_y_preparar_temario(self):
        if not self.contenido_base:
            print("⚠️ Contenido base vacío")
            return
        await self.extractor_temario.extraer_temario(self.contenido_base)
        # Actualizar clasificador semántico con el temario real
        if self.clasificador_semantico:
            self.clasificador_semantico.temario = self.extractor_temario.temas
            print(f"   🔄 Clasificador semántico actualizado con "
                  f"{len(self.extractor_temario.temas)} temas")

    async def indexar_en_neo4j(self, directorio_pdfs: str = DIRECTORIO_PDFS,
                                 imagen_files_extra: Optional[List[str]] = None,
                                 forzar: bool = False):
        # ── Verificar si ya hay datos ────────────────────────────────
        if not forzar:
            try:
                res_chunks = await self.neo4j.run("MATCH (c:Chunk) RETURN count(c) AS n")
                res_imgs   = await self.neo4j.run("MATCH (i:Imagen) RETURN count(i) AS n")
                n_chunks = res_chunks[0]["n"] if res_chunks else 0
                n_imgs   = res_imgs[0]["n"]   if res_imgs   else 0
                if n_chunks > 0 and n_imgs > 0:
                    print(f"✅ Base de datos Neo4j ya poblada ({n_chunks} chunks, {n_imgs} imágenes). Saltando indexación.")
                    print("   (Usá --reindex --force para forzar re-indexación)")
                    return
            except Exception as e:
                print(f"⚠️ No se pudo verificar estado de la BD: {e}")

        print("📄 Indexando chunks de texto en Neo4j...")
        for pdf_path in glob.glob(os.path.join(directorio_pdfs, "*.pdf")):
            fuente = os.path.basename(pdf_path)
            await self.neo4j.upsert_pdf(fuente)
            texto  = self._leer_pdf(pdf_path)
            chunks = self._chunks(texto)
            print(f"  {fuente}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                try:
                    emb       = self._embed_texto(chunk)
                    chunk_id  = f"chunk_{fuente}_{i}"
                    entidades = self.extractor_entidades.extraer_de_texto_sync(chunk)
                    await self.neo4j.upsert_chunk(
                        chunk_id=chunk_id, texto=chunk, fuente=fuente,
                        chunk_idx=i, embedding=emb, entidades=entidades
                    )
                except Exception as e:
                    print(f"  ⚠️ Chunk {i}: {e}")

        print("📸 Indexando imágenes de PDFs en Neo4j...")
        imagenes_pdf = self.extractor_imagenes.extraer_de_directorio(directorio_pdfs)
        for img_info in imagenes_pdf:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                emb_u  = self.uni.embed_image(img_path)
                emb_p  = self.plip.embed_image(img_path)
                
                img_id = f"img_{img_info['fuente_pdf']}_{img_info['pagina']}"
                
                await self.neo4j.upsert_imagen(
                    imagen_id=img_id, path=img_path,
                    fuente=img_info["fuente_pdf"], pagina=img_info["pagina"],
                    ocr_text=img_info.get("ocr_text", ""),
                    texto_pagina=img_info.get("texto_pagina", ""),
                    emb_uni=emb_u.tolist(),
                    emb_plip=emb_p.tolist(),
                    caption=img_info.get("caption", "")
                )
            except Exception as e:
                print(f"  ⚠️ Imagen {img_path}: {e}")

        for img_path in (imagen_files_extra or []):
            if not os.path.exists(img_path):
                continue
            try:

                ocr = ""
                try:
                    ocr = pytesseract.image_to_string(Image.open(img_path)).strip()
                except Exception:
                    pass
                img_id = f"img_extra_{os.path.basename(img_path)}"
                emb_u = self.uni.embed_image(img_path)
                emb_p = self.plip.embed_image(img_path)
                await self.neo4j.upsert_imagen(
                    imagen_id=img_id, path=img_path, fuente=os.path.basename(img_path),
                    pagina=0, ocr_text=ocr[:300], texto_pagina="", emb_uni=emb_u.tolist(), emb_plip=emb_p.tolist()
                )
            except Exception as e:
                print(f"  ❌ Imagen extra {img_path}: {e}")

        await self.neo4j.crear_relaciones_similitud(SIMILAR_IMG_THRESHOLD)
        print("✅ Indexación Neo4j completada")

    # ------------------------------------------------------------------
    # Punto de entrada público
    # ------------------------------------------------------------------

    async def consultar(self, consulta_texto: str,
                         imagen_path: Optional[str] = None,
                         user_id: str = "default_user") -> str:
        """
        La imagen es completamente opcional en cada llamada.
        Si no se pasa, se reutiliza la última imagen activa en memoria.
        """
        imagen_activa       = imagen_path or self.memoria.get_imagen_activa()
        tiene_imagen_activa = self.memoria.tiene_imagen_previa() or bool(imagen_path)

        print(f"\n{'='*70}")
        print(f"🔬 RAG Histología Neo4j v4.2 | umbral={self.SIMILARITY_THRESHOLD}")
        print(f"   Texto:         {consulta_texto}")
        print(f"   Imagen turno:  {imagen_path or 'ninguna'}")
        print(f"   Imagen activa: {imagen_activa or 'ninguna'}")
        print(f"{'='*70}")

        initial_state = AgentState(
            messages=[], consulta_texto=consulta_texto,
            imagen_path=imagen_activa,
            imagen_embedding_uni=None, imagen_embedding_plip=None, texto_embedding=None, contexto_memoria="",
            contenido_base=self.contenido_base, terminos_busqueda="",
            entidades_consulta={"tejidos": [], "estructuras": [], "tinciones": []},
            consulta_busqueda_texto="", consulta_busqueda_visual="",
            resultados_busqueda=[], resultados_validos=[], contexto_documentos="",
            respuesta_final="", trayectoria=[], user_id=user_id, tiempo_inicio=time.time(),
            analisis_visual=None, tiene_imagen=False, imagen_es_nueva=False,
            contexto_suficiente=False, temario=self.extractor_temario.temas,
            tema_valido=True, tema_encontrado=None, imagenes_recuperadas=[],
            imagenes_texto_map={},
            analisis_comparativo=None, estructura_identificada=None,
            similitud_semantica_dominio=0.0, confianza_baja=False,
        )

        config = {
            "configurable": {"thread_id": user_id},
            "run_name":     f"consulta-neo4j-v4.1-{user_id}",
            "tags":         ["rag", "histologia", "neo4j", "imagebind", "v4.1"],
            "metadata": {
                "tiene_imagen_nueva":  imagen_path is not None,
                "tiene_imagen_activa": tiene_imagen_activa,
                "consulta":            consulta_texto[:100],
                "version":             "4.1"
            }
        }
        try:
            final     = await self.compiled_graph.ainvoke(initial_state, config=config)
            respuesta = final["respuesta_final"]
        except Exception as e:
            import traceback; traceback.print_exc()
            respuesta = f"Error: {e}"

        print(f"\n{'='*70}\n📖 RESPUESTA:\n{'='*70}")
        print(respuesta)
        print("="*70)
        return respuesta

    async def cerrar(self):
        if self.neo4j:
            await self.neo4j.close()


# =============================================================================
# MODO INTERACTIVO MEJORADO v4.1
# =============================================================================

async def modo_interactivo(reindex: bool = False, force: bool = False):
    asistente = AsistenteHistologiaNeo4j()
    await asistente.inicializar_componentes()

    print("\n🔄 Leyendo el manual...")
    asistente.procesar_contenido_base(DIRECTORIO_PDFS)

    print("\n📋 Extrayendo temario...")
    await asistente.extraer_y_preparar_temario()
    print(f"   → {len(asistente.extractor_temario.temas)} temas")

    print("\n💾 Indexando en Neo4j...")
    if reindex:
        await asistente.indexar_en_neo4j(DIRECTORIO_PDFS, forzar=force)
    else:
        print("   (Saltando indexación — usar --reindex para forzar)")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  RAG Histología Neo4j v4.2 — Chat Interactivo               ║
╠══════════════════════════════════════════════════════════════╣
║  • Escribí tu pregunta y presioná Enter                     ║
║  • Para subir una imagen: escribí el PATH cuando se pida    ║
║  • La imagen se recuerda entre turnos — no es obligatoria   ║
║  • ✨ Consultas de texto puro — flujo optimizado            ║
║  • Comandos especiales:                                     ║
║      temario       → ver temas disponibles                  ║
║      imagen actual → ver imagen activa en el chat           ║
║      nueva imagen  → limpiar imagen activa                  ║
║      salir         → terminar                               ║
╚══════════════════════════════════════════════════════════════╝
""")

    while True:
        try:
            print("\n" + "─"*60)

            img_activa = asistente.memoria.get_imagen_activa()
            if img_activa:
                print(f"📌 Imagen activa: {os.path.basename(img_activa)} "
                      f"(turno {asistente.memoria.imagen_turno_subida})")

            consulta = input("💬 Vos: ").strip()
            if not consulta:
                continue

            cmd = consulta.lower()

            if cmd in ("salir", "exit", "quit"):
                await asistente.cerrar()
                print("👋 ¡Hasta luego!")
                break

            if cmd == "temario":
                print("\n📚 TEMAS DISPONIBLES:")
                for i, t in enumerate(asistente.extractor_temario.temas, 1):
                    print(f"  {i:3}. {t}")
                continue

            if cmd == "imagen actual":
                if img_activa:
                    print(f"📌 Imagen activa: {img_activa}")
                    print(f"   Subida en turno: {asistente.memoria.imagen_turno_subida}")
                else:
                    print("ℹ️ No hay imagen activa en el chat.")
                continue

            if cmd == "nueva imagen":
                asistente.memoria.set_imagen(None)
                print("🗑️  Imagen activa eliminada. El próximo turno será solo texto.")
                continue

            # Imagen opcional
            imagen_path = None
            img_input   = input("🖼️  Imagen (path o Enter para omitir): ").strip()

            if img_input:
                if os.path.exists(img_input):
                    imagen_path = img_input
                    print(f"✅ Nueva imagen: {imagen_path}")
                else:
                    print(f"⚠️ No encontrada: {img_input} — se usará imagen activa (si la hay)")
            else:
                if img_activa:
                    print(f"♻️  Se usará imagen activa: {os.path.basename(img_activa)}")
                else:
                    print("ℹ️ Sin imagen — consulta solo de texto")

            await asistente.consultar(consulta, imagen_path)

        except KeyboardInterrupt:
            await asistente.cerrar()
            print("\n\n👋 Interrumpido")
            break
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="Indexar en Neo4j (salta si ya hay datos)")
    parser.add_argument("--force", action="store_true", help="Forzar re-indexación aunque haya datos")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    print(f"✅ GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "⚠️ CPU mode")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(DIRECTORIO_IMAGENES, exist_ok=True)
    asyncio.run(modo_interactivo(reindex=args.reindex, force=args.force))