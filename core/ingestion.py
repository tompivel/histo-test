import os
import glob
from typing import List, Optional

from utils.config import DIRECTORIO_PDFS, SIMILAR_IMG_THRESHOLD, userdata
from db.neo4j_client import Neo4jClient
from extractors.pdf import ExtractorImagenesPDF
from extractors.text import ExtractorEntidades, ExtractorTemario
from models.vision import UniWrapper, PlipWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

class PipelineIngestion:
    """
    Módulo ETL (Extract, Transform, Load) para procesar el corpus base y poblar Neo4j.
    Ideal para ser ejecutado de forma estática y aislada en un Google Colab Notebook 
    dejando la inferencia/RAG al core.agent.
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str, device: str = 'cpu'):
        self.db = Neo4jClient(neo4j_uri, neo4j_user, neo4j_pass)
        self.device = device
        
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            api_key=userdata.get("GROQ_API_KEY")
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        
        self.uni = UniWrapper(self.device)
        self.plip = PlipWrapper(self.device)
        self.extractor_imagenes = ExtractorImagenesPDF()
        self.extractor_entidades = ExtractorEntidades(self.llm)
        self.extractor_temario = ExtractorTemario(self.llm)

    async def inicializar(self):
        print("🔄 Conectando a la Base de Datos Neo4j y cargando modelos locales...")
        await self.db.connect()
        await self.db.crear_esquema()
        self.uni.load()
        self.plip.load()

    def _leer_pdf(self, path: str) -> str:
        # Lógica cruda y directa usando PyMuPDF
        import fitz
        try:
            doc = fitz.open(path)
            texto = "\n".join(pagina.get_text() for pagina in doc)
            doc.close()
            return texto
        except Exception as e:
            print(f"⚠️ Error leyendo {path}: {e}")
            return ""

    def _chunks(self, texto: str, size: int = 500) -> List[str]:
        return [texto[i:i+size] for i in range(0, len(texto), size)]

    async def extraer_y_preparar_temario(self, directorio_pdfs: str = DIRECTORIO_PDFS):
        print("📋 Extrayendo e infiriendo el temario general...")
        pdfs = glob.glob(os.path.join(directorio_pdfs, "*.pdf"))
        if not pdfs:
            print("⚠️ No hay PDFs para extraer temario.")
            return

        contenido_base = "\n".join(self._leer_pdf(p) for p in pdfs)
        await self.extractor_temario.extraer_temario(contenido_base)

    async def ejecutar(self, directorio_pdfs: str = DIRECTORIO_PDFS, forzar: bool = False):
        """
        Orquesta o ejecuta el pipeline completo de indexación vectorial (RAG) y Knowledge Graph en Neo4j.
        """
        if not forzar:
            try:
                res = await self.db.run("MATCH (c:Chunk) RETURN count(c) AS n")
                if res and res[0]["n"] > 0:
                    print("✅ Base de datos Neo4j ya poblada. Saltando indexación. (Usa forzar=True)")
                    return
            except Exception as e:
                pass

        await self.extraer_y_preparar_temario(directorio_pdfs)

        print("📄 Indexando chunks de texto en Neo4j...")
        for pdf_path in glob.glob(os.path.join(directorio_pdfs, "*.pdf")):
            fuente = os.path.basename(pdf_path)
            await self.db.upsert_pdf(fuente)
            texto  = self._leer_pdf(pdf_path)
            chunks = self._chunks(texto)
            print(f"  {fuente}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                try:
                    emb       = self.embeddings.embed_query(chunk)
                    chunk_id  = f"chunk_{fuente}_{i}"
                    entidades = self.extractor_entidades.extraer_de_texto_sync(chunk)
                    await self.db.upsert_chunk(
                        chunk_id=chunk_id, texto=chunk, fuente=fuente,
                        chunk_idx=i, embedding=emb, entidades=entidades
                    )
                except Exception as e:
                    print(f"  ⚠️ Chunk {i}: {e}")

        print("📸 Extrayendo e Indexando imágenes de PDFs (Vision Models)...")
        imagenes_pdf = self.extractor_imagenes.extraer_de_directorio(directorio_pdfs)
        for img_info in imagenes_pdf:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                emb_u  = self.uni.embed_image(img_path)
                emb_p  = self.plip.embed_image(img_path)
                
                img_id = f"img_{img_info['fuente_pdf']}_{img_info['pagina']}"
                
                await self.db.upsert_imagen(
                    imagen_id=img_id, path=img_path,
                    fuente=img_info["fuente_pdf"], pagina=img_info["pagina"],
                    ocr_text=img_info.get("ocr_text", ""),
                    texto_pagina=img_info.get("texto_pagina", ""),
                    emb_uni=emb_u.tolist(),
                    emb_plip=emb_p.tolist()
                )
            except Exception as e:
                print(f"  ⚠️ Imagen {img_path}: {e}")

        print("🔗 Construyendo relaciones de similitud visual K-NN...")
        await self.db.crear_relaciones_similitud(SIMILAR_IMG_THRESHOLD)
        print("✅ Indexación ETL completada satisfactoriamente.")

    async def cerrar(self):
        await self.db.close()
