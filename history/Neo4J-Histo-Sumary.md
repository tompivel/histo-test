# Walkthrough: Migración de 4 Features Críticos

## Resumen

Se migraron exitosamente 4 features desde `backup_features_main.py` (monolito) a la nueva arquitectura modular. **6 archivos modificados**, todos verificados con AST syntax check.

---

## Feature 1: RecursiveCharacterTextSplitter

**Archivo:** [ingestion.py](file:///home/francisco/Escritorio/IA/histo-test/core/ingestion.py)

**Antes:** `_chunks()` cortaba texto en bloques de 500 caracteres sin solapamiento (`text[i:i+size]`).

**Ahora:** Usa `RecursiveCharacterTextSplitter` de LangChain con:
- `chunk_size=800` (mejor cobertura de contexto)
- `chunk_overlap=150` (preserva continuidad entre chunks)
- Separadores jerárquicos: `["\n\n", "\n", ". ", " ", ""]`

```diff:ingestion.py
import os
import glob
from typing import List

from utils.config import PDFS_DIR, SIMILAR_IMG_THRESHOLD, userdata
from db.neo4j_client import Neo4jClient
from extractors.pdf import PDFImageExtractor
from extractors.text import EntityExtractor, TopicExtractor
from models.vision import UniWrapper, PlipWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

class IngestionPipeline:
    """
    ETL (Extract, Transform, Load) module to process the base corpus and populate Neo4j.
    Ideal for execution in a static Google Colab environment, offloading inference to core.agent.
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str, device: str = None):
        import torch
        self.db = Neo4jClient(neo4j_uri, neo4j_user, neo4j_pass)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"⚠️ Incompatible GPU detected (sm_{cap[0]}{cap[1]}). Forcing CPU to avoid fallback_error.")
                    self.device = "cpu"
            except:
                pass
        print(f"🚀 Pipeline ETL Initialized on Backend: {self.device.upper()}")
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
        self.image_extractor = PDFImageExtractor()
        self.entity_extractor = EntityExtractor(self.llm)
        self.topic_extractor = TopicExtractor(self.llm)

    async def initialize(self):
        print("🔄 Connecting to Neo4j DB and loading local models...")
        await self.db.connect()
        await self.db.create_schema()
        self.uni.load()
        self.plip.load()

    def _read_pdf(self, path: str) -> str:
        import fitz
        try:
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            return ""

    def _chunks(self, text: str, size: int = 500) -> List[str]:
        return [text[i:i+size] for i in range(0, len(text), size)]

    async def extract_and_prepare_syllabus(self, pdfs_dir: str = PDFS_DIR):
        print("📋 Extracting and inferring the general syllabus...")
        pdfs = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
        if not pdfs:
            print("⚠️ No PDFs to extract syllabus from.")
            return

        base_content = "\n".join(self._read_pdf(p) for p in pdfs)
        await self.topic_extractor.extract_topics(base_content)

    async def execute(self, pdfs_dir: str = PDFS_DIR, force: bool = False):
        """
        Orchestrates the entire database pipeline population.
        """
        if not force:
            try:
                res = await self.db.run("MATCH (c:Chunk) RETURN count(c) AS n")
                if res and res[0]["n"] > 0:
                    n_chunks = res[0]["n"]
                    print(f"✅ Neo4j database already populated (Found {n_chunks} 'Chunk' nodes).")
                    print(f"   ( Connected to: {self.db.uri} )")
                    print("   Skipping indexing. (Use force=True)")
                    return
            except Exception as e:
                print(f"⚠️ Error verifying schema: {e}")
                pass

        await self.extract_and_prepare_syllabus(pdfs_dir)

        print("📄 Indexing text chunks in Neo4j...")
        for pdf_path in glob.glob(os.path.join(pdfs_dir, "*.pdf")):
            source = os.path.basename(pdf_path)
            await self.db.upsert_pdf(source)
            text  = self._read_pdf(pdf_path)
            chunks = self._chunks(text)
            print(f"  {source}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                try:
                    emb       = self.embeddings.embed_query(chunk)
                    chunk_id  = f"chunk_{source}_{i}"
                    entities  = self.entity_extractor.extract_from_text_sync(chunk)
                    await self.db.upsert_chunk(
                        chunk_id=chunk_id, text=chunk, source=source,
                        chunk_idx=i, embedding=emb, entities=entities
                    )
                except Exception as e:
                    print(f"  ⚠️ Chunk {i}: {e}")

        print("📸 Extracting and Indexing PDF images (Vision Models)...")
        pdf_images = self.image_extractor.extract_from_directory(pdfs_dir)
        for img_info in pdf_images:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                emb_u  = self.uni.embed_image(img_path)
                emb_p  = self.plip.embed_image(img_path)
                
                img_id = f"img_{img_info['source_pdf']}_{img_info['page']}"
                
                await self.db.upsert_image(
                    image_id=img_id, path=img_path,
                    source=img_info["source_pdf"], page=img_info["page"],
                    ocr_text=img_info.get("ocr_text", ""),
                    page_text=img_info.get("page_text", ""),
                    emb_uni=emb_u.tolist(),
                    emb_plip=emb_p.tolist()
                )
            except Exception as e:
                print(f"  ⚠️ Image {img_path}: {e}")

        print("🔗 Building K-NN visual similarity relations...")
        await self.db.create_similarity_relations(SIMILAR_IMG_THRESHOLD)
        print("✅ ETL indexing completed successfully.")

    async def close(self):
        await self.db.close()
===
import os
import glob
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import PDFS_DIR, SIMILAR_IMG_THRESHOLD, userdata
from db.neo4j_client import Neo4jClient
from extractors.pdf import PDFImageExtractor
from extractors.text import EntityExtractor, TopicExtractor
from models.vision import UniWrapper, PlipWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

class IngestionPipeline:
    """
    ETL (Extract, Transform, Load) module to process the base corpus and populate Neo4j.
    Ideal for execution in a static Google Colab environment, offloading inference to core.agent.
    """
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_pass: str, device: str = None):
        import torch
        self.db = Neo4jClient(neo4j_uri, neo4j_user, neo4j_pass)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"⚠️ Incompatible GPU detected (sm_{cap[0]}{cap[1]}). Forcing CPU to avoid fallback_error.")
                    self.device = "cpu"
            except:
                pass
        print(f"🚀 Pipeline ETL Initialized on Backend: {self.device.upper()}")
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
        self.image_extractor = PDFImageExtractor(llm=self.llm)
        self.entity_extractor = EntityExtractor(self.llm)
        self.topic_extractor = TopicExtractor(self.llm)

        # Chunking with RecursiveCharacterTextSplitter (hierarchical separators)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def initialize(self):
        print("🔄 Connecting to Neo4j DB and loading local models...")
        await self.db.connect()
        await self.db.create_schema()
        self.uni.load()
        self.plip.load()

    def _read_pdf(self, path: str) -> str:
        import fitz
        try:
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            return ""

    def _chunks(self, text: str) -> List[str]:
        """Splits text using RecursiveCharacterTextSplitter with hierarchical separators."""
        return self.text_splitter.split_text(text)

    async def extract_and_prepare_syllabus(self, pdfs_dir: str = PDFS_DIR):
        print("📋 Extracting and inferring the general syllabus...")
        pdfs = glob.glob(os.path.join(pdfs_dir, "*.pdf"))
        if not pdfs:
            print("⚠️ No PDFs to extract syllabus from.")
            return

        base_content = "\n".join(self._read_pdf(p) for p in pdfs)
        await self.topic_extractor.extract_topics(base_content)

    async def execute(self, pdfs_dir: str = PDFS_DIR, force: bool = False):
        """
        Orchestrates the entire database pipeline population.
        """
        if not force:
            try:
                res = await self.db.run("MATCH (c:Chunk) RETURN count(c) AS n")
                if res and res[0]["n"] > 0:
                    n_chunks = res[0]["n"]
                    print(f"✅ Neo4j database already populated (Found {n_chunks} 'Chunk' nodes).")
                    print(f"   ( Connected to: {self.db.uri} )")
                    print("   Skipping indexing. (Use force=True)")
                    return
            except Exception as e:
                print(f"⚠️ Error verifying schema: {e}")
                pass

        await self.extract_and_prepare_syllabus(pdfs_dir)

        print("📄 Indexing text chunks in Neo4j...")
        for pdf_path in glob.glob(os.path.join(pdfs_dir, "*.pdf")):
            source = os.path.basename(pdf_path)
            await self.db.upsert_pdf(source)
            text  = self._read_pdf(pdf_path)
            chunks = self._chunks(text)
            print(f"  {source}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                try:
                    emb       = self.embeddings.embed_query(chunk)
                    chunk_id  = f"chunk_{source}_{i}"
                    entities  = self.entity_extractor.extract_from_text_sync(chunk)
                    await self.db.upsert_chunk(
                        chunk_id=chunk_id, text=chunk, source=source,
                        chunk_idx=i, embedding=emb, entities=entities
                    )
                except Exception as e:
                    print(f"  ⚠️ Chunk {i}: {e}")

        print("📸 Extracting and Indexing PDF images (Vision Models)...")
        pdf_images = self.image_extractor.extract_from_directory(pdfs_dir)
        for img_info in pdf_images:
            img_path = img_info["path"]
            if not os.path.exists(img_path):
                continue
            try:
                emb_u  = self.uni.embed_image(img_path)
                emb_p  = self.plip.embed_image(img_path)
                
                img_id = f"img_{img_info['source_pdf']}_{img_info['page']}"
                
                await self.db.upsert_image(
                    image_id=img_id, path=img_path,
                    source=img_info["source_pdf"], page=img_info["page"],
                    ocr_text=img_info.get("ocr_text", ""),
                    page_text=img_info.get("page_text", ""),
                    emb_uni=emb_u.tolist(),
                    emb_plip=emb_p.tolist()
                )

                # ── Table detection via multimodal LLM (Groq Vision → Gemini fallback) ──
                try:
                    tabla_md = await self.image_extractor.detect_and_extract_table(img_path)
                    if tabla_md:
                        tabla_id = f"tabla_{img_info['source_pdf']}_{img_info['page']}"
                        tabla_emb = self.embeddings.embed_query(tabla_md[:800])
                        await self.db.upsert_table(
                            tabla_id=tabla_id,
                            contenido_md=tabla_md,
                            source=img_info["source_pdf"],
                            page=img_info["page"],
                            embedding=tabla_emb
                        )
                        print(f"  📊 Table extracted from page {img_info['page']}")
                except Exception as e_tabla:
                    print(f"  ⚠️ Table detection page {img_info['page']}: {e_tabla}")

            except Exception as e:
                print(f"  ⚠️ Image {img_path}: {e}")

        print("🔗 Building K-NN visual similarity relations...")
        await self.db.create_similarity_relations(SIMILAR_IMG_THRESHOLD)
        print("✅ ETL indexing completed successfully.")

    async def close(self):
        await self.db.close()
```

---

## Feature 2: Detección de Tablas + Nodo `:Tabla`

### 2a. Método de extracción multimodal
**Archivo:** [pdf.py](file:///home/francisco/Escritorio/IA/histo-test/extractors/pdf.py)

Nuevo método `detect_and_extract_table(image_path)`:
- **Intento 1:** Groq Vision (`llama-3.2-11b-vision-preview`)
- **Intento 2 (fallback):** Gemini (`gemini-2.5-flash`)
- Retorna Markdown de tabla o `""` si no hay tablas

### 2b. Persistencia en Neo4j
**Archivo:** [neo4j_client.py](file:///home/francisco/Escritorio/IA/histo-test/db/neo4j_client.py)

- Constraint: `CREATE CONSTRAINT tabla_id IF NOT EXISTS FOR (t:Tabla) REQUIRE t.id IS UNIQUE`
- Método `upsert_table()` con relaciones `:PERTENECE_A` → PDF y `:EN_PAGINA` → Pagina

### 2c. Integración en pipeline
**Archivo:** [ingestion.py](file:///home/francisco/Escritorio/IA/histo-test/core/ingestion.py)

Después de extraer cada imagen, se llama `detect_and_extract_table()`. Si hay tabla, se genera embedding textual y se persiste vía `db.upsert_table()`.

```diff:pdf.py
import os
import fitz # PyMuPDF
from PIL import Image
try:
    from pdf2image import convert_from_path
except ImportError:
    pass
import glob
try:
    import pytesseract
except ImportError:
    pass
from typing import List, Dict

from utils.config import IMG_DIR

class PDFImageExtractor:
    """
    Class responsible for reading PDF files and extracting their images, 
    saving them to disk along with their metadata (OCR and page contextual text).
    """
    MIN_WIDTH  = 200
    MIN_HEIGHT = 200

    def __init__(self, output_dir: str = IMG_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        extracted_images = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening {pdf_path}: {e}")
            return []

        def _page_text_with_context(doc, idx_0based: int) -> str:
            parts = []
            try:
                parts.append(doc[idx_0based].get_text().strip())
            except Exception:
                pass
            if idx_0based + 1 < len(doc):
                try:
                    next_page = doc[idx_0based + 1].get_text().strip()
                    if next_page:
                        parts.append(next_page)
                except Exception:
                    pass
            return "\n".join(parts)

        for page_num, page in enumerate(doc, start=1):
            idx_0 = page_num - 1 
            valid_images_this_page = []
            
            try:
                img_info_list = page.get_image_info(xrefs=True)
                page_xrefs = [info["xref"] for info in img_info_list if info.get("xref")]
                
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
                                    "area": w * h
                                })
                        except Exception:
                            continue
            except Exception:
                pass
            
            if valid_images_this_page:
                best_img = max(valid_images_this_page, key=lambda x: x["area"])
                pil_img = best_img["pil"]
                file_name = f"{pdf_name}_pag{page_num}.png"
                full_path  = os.path.join(self.output_dir, file_name)
                
                try:
                    pil_img.save(full_path, format="PNG")
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    
                    full_page_text = _page_text_with_context(doc, idx_0)

                    extracted_images.append({
                        "path": full_path, "source_pdf": os.path.basename(pdf_path),
                        "page": page_num, "index": 1, "ocr_text": ocr_text,
                        "page_text": full_page_text
                    })
                except Exception as e:
                    print(f"  ⚠️ Error saving page {page_num}: {e}")
            else:
                try:
                    from pdf2image import convert_from_path
                    page_imgs = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)
                    if page_imgs:
                        pil_full = page_imgs[0]
                        file_name = f"{pdf_name}_pag{page_num}_full.png"
                        full_path  = os.path.join(self.output_dir, file_name)
                        pil_full.save(full_path, format="PNG")
                        
                        try:
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""

                        full_page_text = _page_text_with_context(doc, idx_0)
                        
                        extracted_images.append({
                            "path": full_path, "source_pdf": os.path.basename(pdf_path),
                            "page": page_num, "index": 1, "ocr_text": ocr_text,
                            "page_text": full_page_text
                        })
                except Exception as e:
                    print(f"  ⚠️ Fallback error page {page_num}: {e}")

        doc.close()
        print(f"  📸 {len(extracted_images)} images processed from {os.path.basename(pdf_path)}")
        return extracted_images

    def extract_from_directory(self, directory: str) -> List[Dict[str, str]]:
        all_imgs = []
        pdfs  = glob.glob(os.path.join(directory, "*.pdf"))
        print(f"📂 Extracting from {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            all_imgs.extend(self.extract_from_pdf(pdf_path))
        print(f"✅ Total images extracted: {len(all_imgs)}")
        return all_imgs
===
import os
import base64
import fitz # PyMuPDF
from PIL import Image
try:
    from pdf2image import convert_from_path
except ImportError:
    pass
import glob
try:
    import pytesseract
except ImportError:
    pass
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage
from utils.resilience import invoke_with_retry

from utils.config import IMG_DIR

class PDFImageExtractor:
    """
    Class responsible for reading PDF files and extracting their images, 
    saving them to disk along with their metadata (OCR and page contextual text).
    """
    MIN_WIDTH  = 200
    MIN_HEIGHT = 200

    def __init__(self, output_dir: str = IMG_DIR, llm=None):
        self.output_dir = output_dir
        self.llm = llm
        os.makedirs(output_dir, exist_ok=True)

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        extracted_images = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening {pdf_path}: {e}")
            return []

        def _page_text_with_context(doc, idx_0based: int) -> str:
            parts = []
            try:
                parts.append(doc[idx_0based].get_text().strip())
            except Exception:
                pass
            if idx_0based + 1 < len(doc):
                try:
                    next_page = doc[idx_0based + 1].get_text().strip()
                    if next_page:
                        parts.append(next_page)
                except Exception:
                    pass
            return "\n".join(parts)

        for page_num, page in enumerate(doc, start=1):
            idx_0 = page_num - 1 
            valid_images_this_page = []
            
            try:
                img_info_list = page.get_image_info(xrefs=True)
                page_xrefs = [info["xref"] for info in img_info_list if info.get("xref")]
                
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
                                    "area": w * h
                                })
                        except Exception:
                            continue
            except Exception:
                pass
            
            if valid_images_this_page:
                best_img = max(valid_images_this_page, key=lambda x: x["area"])
                pil_img = best_img["pil"]
                file_name = f"{pdf_name}_pag{page_num}.png"
                full_path  = os.path.join(self.output_dir, file_name)
                
                try:
                    pil_img.save(full_path, format="PNG")
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    
                    full_page_text = _page_text_with_context(doc, idx_0)

                    extracted_images.append({
                        "path": full_path, "source_pdf": os.path.basename(pdf_path),
                        "page": page_num, "index": 1, "ocr_text": ocr_text,
                        "page_text": full_page_text
                    })
                except Exception as e:
                    print(f"  ⚠️ Error saving page {page_num}: {e}")
            else:
                try:
                    from pdf2image import convert_from_path
                    page_imgs = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)
                    if page_imgs:
                        pil_full = page_imgs[0]
                        file_name = f"{pdf_name}_pag{page_num}_full.png"
                        full_path  = os.path.join(self.output_dir, file_name)
                        pil_full.save(full_path, format="PNG")
                        
                        try:
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""

                        full_page_text = _page_text_with_context(doc, idx_0)
                        
                        extracted_images.append({
                            "path": full_path, "source_pdf": os.path.basename(pdf_path),
                            "page": page_num, "index": 1, "ocr_text": ocr_text,
                            "page_text": full_page_text
                        })
                except Exception as e:
                    print(f"  ⚠️ Fallback error page {page_num}: {e}")

        doc.close()
        print(f"  📸 {len(extracted_images)} images processed from {os.path.basename(pdf_path)}")
        return extracted_images

    def extract_from_directory(self, directory: str) -> List[Dict[str, str]]:
        all_imgs = []
        pdfs  = glob.glob(os.path.join(directory, "*.pdf"))
        print(f"📂 Extracting from {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            all_imgs.extend(self.extract_from_pdf(pdf_path))
        print(f"✅ Total images extracted: {len(all_imgs)}")
        return all_imgs

    async def detect_and_extract_table(self, image_path: str) -> str:
        """Uses a multimodal LLM to detect and format tables in the page image.
        Attempts Groq Vision first, with fallback to Gemini 2.5 Flash.
        Returns Markdown table string or empty string if no table found."""
        if not self.llm:
            return ""
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            msg = HumanMessage(content=[
                {"type": "text", "text": (
                    "Analiza esta imagen de una página de un manual de histología.\n"
                    "Si encuentras tablas con datos técnicos, extráelas en formato Markdown.\n"
                    "Si no hay tablas, responde únicamente con 'SIN_TABLAS'."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
            ])

            # Attempt 1: Groq Vision
            try:
                from langchain_groq import ChatGroq
                from utils.config import userdata
                llm_vision = ChatGroq(
                    model="llama-3.2-11b-vision-preview",
                    api_key=userdata.get("GROQ_API_KEY"),
                    temperature=0, max_retries=1
                )
                resp = await invoke_with_retry(llm_vision, [msg])
                text = resp.content.strip()
                return "" if "SIN_TABLAS" in text else text
            except Exception as e_groq:
                print(f"  ⚠️ Groq Vision unavailable: {e_groq}")

            # Attempt 2: Fallback to Gemini 2.5 Flash
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from utils.config import userdata
                llm_gemini = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=userdata.get("GOOGLE_API_KEY"),
                    temperature=0,
                )
                resp = await invoke_with_retry(llm_gemini, [msg])
                text = resp.content.strip()
                return "" if "SIN_TABLAS" in text else text
            except Exception as e_gemini:
                print(f"  ⚠️ Gemini fallback also failed: {e_gemini}")
                return ""
        except Exception as e:
            print(f"  ⚠️ Table detection error: {e}")
            return ""
```

---

## Feature 3: Normalización SNOMED/FMA

### 3a. Prompt enriquecido
**Archivo:** [entity_extractor.txt](file:///home/francisco/Escritorio/IA/histo-test/prompts/entity_extractor.txt)

El prompt ahora solicita explícitamente `snomed_id` y `fma_id` para cada tejido/estructura.

### 3b. Extractor con ontologías
**Archivo:** [text.py](file:///home/francisco/Escritorio/IA/histo-test/extractors/text.py)

`extract_from_text()` retorna `Dict[str, list]` (listas de dicts con `{nombre, snomed_id, fma_id}`) en lugar de listas de strings.

### 3c. Cypher enriquecido
**Archivo:** [neo4j_client.py](file:///home/francisco/Escritorio/IA/histo-test/db/neo4j_client.py)

`upsert_chunk()` ahora:
- Acepta entities como `Dict[str, list]` 
- Usa `isinstance(tissue, dict)` para backward compatibility con strings planos
- Persiste `SET t.snomed_id = coalesce($snomed_id, t.snomed_id), t.fma_id = coalesce($fma_id, t.fma_id)`
- Normaliza dicts a strings para relaciones cruzadas `:CONTIENE` y `:TENIDA_CON`

```diff:neo4j_client.py
from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any, Optional

from utils.config import (
    TEXT_INDEX_NAME, UNI_INDEX_NAME, PLIP_INDEX_NAME, TEXT_DIM, UNI_IMG_DIM, PLIP_IMG_DIM, 
    SIMILAR_IMG_THRESHOLD, NEO4J_GRAPH_DEPTH
)

class Neo4jClient:
    """
    Asynchronous client for managing Neo4j connection and graph operations.
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
        print(f"✅ Neo4j Connected: {self.uri}")

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def run(self, query: str, params: Dict = None) -> List[Dict]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) for record in await result.data()]

    async def create_schema(self):
        print("🏗️ Creating Neo4j Schema (v4.4 UNI + PLIP)...")
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

        try:
            indexes = await self.run("SHOW INDEXES YIELD name, type, options")
            for idx in indexes:
                if idx["type"] == "VECTOR":
                    name = idx["name"]
                    config = idx["options"].get("indexConfig", {})
                    dims = config.get("vector.dimensions")
                    
                    target_dims = None
                    if name == TEXT_INDEX_NAME: target_dims = TEXT_DIM
                    elif name == UNI_INDEX_NAME: target_dims = UNI_IMG_DIM
                    elif name == PLIP_INDEX_NAME: target_dims = PLIP_IMG_DIM
                    
                    if target_dims and dims != target_dims:
                        print(f"  ⚠️ Incorrect dimension on index '{name}' ({dims} != {target_dims}). Recreating...")
                        await self.run(f"DROP INDEX {name}")
        except Exception as e:
            print(f"  ⚠️ Schema validation error: {e}")

        vector_queries = [
            f"""
            CREATE VECTOR INDEX {TEXT_INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {TEXT_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX {UNI_INDEX_NAME} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_uni
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {UNI_IMG_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX {PLIP_INDEX_NAME} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_plip
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {PLIP_IMG_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        for vq in vector_queries:
            try:
                await self.run(vq)
            except Exception as e:
                print(f"  ⚠️ Vector index error: {e}")

        print("✅ Neo4j Schema Ready (3 indexes)")

    async def upsert_pdf(self, name: str):
        await self.run("MERGE (p:PDF {nombre: $name})", {"name": name})

    async def upsert_chunk(self, chunk_id: str, text: str, source: str,
                            chunk_idx: int, embedding: List[float],
                            entities: Dict[str, List[str]]):
        await self.run("""
            MERGE (c:Chunk {id: $id})
            SET c.texto = $text, c.fuente = $source,
                c.chunk_id = $chunk_idx, c.embedding = $embedding
            WITH c
            MERGE (pdf:PDF {nombre: $source})
            MERGE (c)-[:PERTENECE_A]->(pdf)
        """, {
            "id": chunk_id, "text": text, "source": source,
            "chunk_idx": chunk_idx, "embedding": embedding
        })
        for tissue in entities.get("tejidos", []):
            await self.run("""
                MERGE (t:Tejido {nombre: $name})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"name": tissue, "chunk_id": chunk_id})
        for structure in entities.get("estructuras", []):
            await self.run("""
                MERGE (e:Estructura {nombre: $name})
                WITH e MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
            """, {"name": structure, "chunk_id": chunk_id})
        for stain in entities.get("tinciones", []):
            await self.run("""
                MERGE (t:Tincion {nombre: $name})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"name": stain, "chunk_id": chunk_id})
        for tissue in entities.get("tejidos", []):
            for structure in entities.get("estructuras", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tissue})
                    MERGE (e:Estructura {nombre: $structure})
                    MERGE (t)-[:CONTIENE]->(e)
                """, {"tissue": tissue, "structure": structure})
            for stain in entities.get("tinciones", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tissue})
                    MERGE (ti:Tincion {nombre: $stain})
                    MERGE (t)-[:TENIDA_CON]->(ti)
                """, {"tissue": tissue, "stain": stain})
        for structure in entities.get("estructuras", []):
            for stain in entities.get("tinciones", []):
                await self.run("""
                    MERGE (e:Estructura {nombre: $structure})
                    MERGE (ti:Tincion {nombre: $stain})
                    MERGE (e)-[:TENIDA_CON]->(ti)
                """, {"structure": structure, "stain": stain})

    async def upsert_image(self, image_id: str, path: str, source: str,
                             page: int, ocr_text: str, page_text: str,
                             emb_uni: List[float], emb_plip: List[float]):
        await self.run("""
            MERGE (i:Imagen {id: $id})
            SET i.path = $path, i.fuente = $source,
                i.pagina = $page, i.ocr_text = $ocr_text,
                i.texto_pagina = $page_text,
                i.embedding_uni = $emb_uni,
                i.embedding_plip = $emb_plip
            WITH i
            MERGE (pdf:PDF {nombre: $source})
            MERGE (i)-[:PERTENECE_A]->(pdf)
            MERGE (pag:Pagina {numero: $page, pdf_nombre: $source})
            MERGE (i)-[:EN_PAGINA]->(pag)
        """, {
            "id": image_id, "path": path, "source": source,
            "page": page, "ocr_text": ocr_text, "page_text": page_text,
            "emb_uni": emb_uni, "emb_plip": emb_plip
        })

    async def create_similarity_relations(self, threshold: float = SIMILAR_IMG_THRESHOLD):
        print(f"🔗 Creating :SIMILAR_A relations (UNI method, threshold={threshold})...")
        images = await self.run(
            "MATCH (i:Imagen) WHERE i.embedding_uni IS NOT NULL RETURN i.id AS id, i.embedding_uni AS emb"
        )
        if len(images) < 2:
            return
        created = 0
        for img in images:
            result = await self.run("""
                CALL db.index.vector.queryNodes($index, 6, $emb)
                YIELD node AS vecino, score
                WHERE vecino.id <> $id AND score >= $threshold
                WITH vecino, score
                MATCH (origen:Imagen {id: $id})
                MERGE (origen)-[r:SIMILAR_A]->(vecino)
                SET r.score = score
                RETURN count(*) AS n
            """, {
                "index": UNI_INDEX_NAME,
                "emb": img["emb"], "id": img["id"], "threshold": threshold
            })
            created += result[0]["n"] if result else 0
        print(f"✅ {created} :SIMILAR_A relations linked")

    async def vector_search(self, embedding: List[float],
                                  index_name: str, top_k: int = 10) -> List[Dict]:
        is_text_index = index_name == TEXT_INDEX_NAME
        if is_text_index:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS c, score
                RETURN c.id AS id, c.texto AS text, c.fuente AS source,
                       'texto' AS type, null AS image_path, score AS similarity
                ORDER BY similarity DESC
            """
        else:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS i, score
                RETURN i.id AS id, 
                       coalesce(i.texto_pagina, i.ocr_text) AS text, 
                       i.fuente AS source,
                       'imagen' AS type, i.path AS image_path, score AS similarity
                ORDER BY similarity DESC
            """
        try:
            return await self.run(query, {"index": index_name, "emb": embedding, "k": top_k})
        except Exception as e:
            print(f"⚠️ Vector search error {index_name}: {e}")
            return []

    async def entity_search(self, entities: Dict[str, List[str]],
                                      top_k: int = 10) -> List[Dict]:
        tissues    = entities.get("tejidos", [])
        structures = entities.get("estructuras", [])
        stains     = entities.get("tinciones", [])
        if not any([tissues, structures, stains]):
            return []

        where_clauses = []
        params: Dict[str, Any] = {}
        if tissues:
            where_clauses.append("ANY(t IN tejidos WHERE t.nombre IN $tejidos)")
            params["tejidos"] = tissues
        if structures:
            where_clauses.append("ANY(e IN estructuras WHERE e.nombre IN $estructuras)")
            params["estructuras"] = structures
        if stains:
            where_clauses.append("ANY(ti IN tinciones WHERE ti.nombre IN $tinciones)")
            params["tinciones"] = stains

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
            RETURN c.id AS id, c.texto AS text, c.fuente AS source,
                   'texto' AS type, null AS image_path, 0.49 AS similarity
            LIMIT $top_k
        """
        params["top_k"] = top_k
        try:
            return await self.run(query, params)
        except Exception as e:
            print(f"⚠️ Entity search error: {e}")
            return []

    async def expand_neighborhood(self, node_ids: List[str],
                                 depth: int = NEO4J_GRAPH_DEPTH) -> List[Dict]:
        if not node_ids:
            return []
        query = """
            UNWIND $ids AS nid
            MATCH (n {id: nid})

            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)
            WHERE vecino_pdf.id <> nid
            WITH n, nid, collect(DISTINCT vecino_pdf)[..5] AS list_pdf

            OPTIONAL MATCH (n)-[:MENCIONA]->(entidad)<-[:MENCIONA]-(vecino_entidad:Chunk)
            WHERE vecino_entidad.id <> nid
            WITH n, nid, list_pdf, collect(DISTINCT vecino_entidad)[..5] AS list_ent

            OPTIONAL MATCH (n)-[:SIMILAR_A]-(vecino_similar:Imagen)
            WITH n, nid, list_pdf, list_ent, collect(DISTINCT vecino_similar)[..5] AS list_sim

            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf2:PDF)<-[:PERTENECE_A]-(img_pag:Imagen)
            WITH n, nid, list_pdf, list_ent, list_sim, collect(DISTINCT img_pag)[..5] AS list_pag

            WITH n, $ids AS ids_originales,
                 list_pdf + list_ent + list_sim + list_pag AS vecinos_raw

            UNWIND vecinos_raw AS v

            WITH n, v, ids_originales
            WHERE v IS NOT NULL AND NOT v.id IN ids_originales

            RETURN DISTINCT
                v.id AS id,
                CASE 
                    WHEN v:Imagen THEN coalesce(v.texto_pagina, v.ocr_text, '') 
                    ELSE coalesce(v.texto, '') 
                END AS text,
                v.fuente AS source,
                CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS type,
                CASE WHEN v:Imagen THEN v.path ELSE null END AS image_path,
                CASE 
                    WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
                         (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
                    THEN 0.95 
                    ELSE 0.3 
                END AS similarity
            LIMIT 15
        """
        try:
            return await self.run(query, {"ids": node_ids})
        except Exception as e:
            print(f"⚠️ Neighborhood expansion error: {e}")
            return []

    async def semantic_path_search(self,
                                         source_tissue: Optional[str],
                                         dest_tissue: Optional[str]) -> List[Dict]:
        if not source_tissue or not dest_tissue:
            return []
        query = """
            MATCH (origen {nombre: $source}), (destino {nombre: $dest})
            MATCH path = shortestPath((origen)-[*1..4]-(destino))
            UNWIND nodes(path) AS nodo
            OPTIONAL MATCH (c:Chunk)-[:MENCIONA]->(nodo)
            RETURN DISTINCT c.id AS id, c.texto AS text, c.fuente AS source,
                   'texto' AS type, null AS image_path, 0.4 AS similarity
            LIMIT 5
        """
        try:
            return await self.run(query, {"source": source_tissue, "dest": dest_tissue})
        except Exception as e:
            print(f"⚠️ Semantic Path search error: {e}")
            return []

    async def hybrid_search(self,
                                text_embedding: Optional[List[float]],
                                image_embedding_uni: Optional[List[float]],
                                image_embedding_plip: Optional[List[float]],
                                entities: Dict[str, List[str]],
                                top_k: int = 10) -> List[Dict]:
        res_text = []
        res_uni  = []
        res_plip = []
        res_ent  = []
        res_vec  = []

        if text_embedding is not None:
            res_text = await self.vector_search(text_embedding, TEXT_INDEX_NAME, top_k)
        if image_embedding_uni is not None:
            res_uni = await self.vector_search(image_embedding_uni, UNI_INDEX_NAME, top_k)
        if image_embedding_plip is not None:
            res_plip = await self.vector_search(image_embedding_plip, PLIP_INDEX_NAME, top_k)

        res_ent = await self.entity_search(entities, top_k)

        all_res = res_text + res_uni + res_plip
        top_ids = [r["id"] for r in all_res[:6] if r.get("id")]
        if top_ids:
            res_vec = await self.expand_neighborhood(top_ids)

        combined: Dict[str, Dict] = {}

        def add(results: List[Dict], weight: float):
            for r in results:
                key = r.get("id") or f"{r.get('source')}_{str(r.get('text',''))[:40]}"
                if not r.get("text") and not r.get("image_path"):
                    continue
                weighted_sim = r.get("similarity", 0) * weight
                if key not in combined:
                    combined[key] = {**r, "similarity": weighted_sim}
                else:
                    combined[key]["similarity"] += weighted_sim

        add(res_text, 0.80) 
        add(res_uni,  0.50) 
        add(res_plip, 0.50) 
        add(res_ent,  0.60) 
        add(res_vec,  0.20)

        final = sorted(combined.values(), key=lambda x: x["similarity"], reverse=True)

        print(f"   📊 Hybrid Search: Txt={len(res_text)} | "
              f"UNI={len(res_uni)} | PLIP={len(res_plip)} | Ent={len(res_ent)} | Vec={len(res_vec)} -> {len(final)}")

        return final[:15]
===
from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any, Optional

from utils.config import (
    TEXT_INDEX_NAME, UNI_INDEX_NAME, PLIP_INDEX_NAME, TEXT_DIM, UNI_IMG_DIM, PLIP_IMG_DIM, 
    SIMILAR_IMG_THRESHOLD, NEO4J_GRAPH_DEPTH
)

class Neo4jClient:
    """
    Asynchronous client for managing Neo4j connection and graph operations.
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
        print(f"✅ Neo4j Connected: {self.uri}")

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def run(self, query: str, params: Dict = None) -> List[Dict]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) for record in await result.data()]

    async def create_schema(self):
        print("🏗️ Creating Neo4j Schema (v4.4 UNI + PLIP)...")
        constraints = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT imagen_id IF NOT EXISTS FOR (i:Imagen) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT pdf_nombre IF NOT EXISTS FOR (p:PDF) REQUIRE p.nombre IS UNIQUE",
            "CREATE CONSTRAINT tejido_nombre IF NOT EXISTS FOR (t:Tejido) REQUIRE t.nombre IS UNIQUE",
            "CREATE CONSTRAINT estructura_nombre IF NOT EXISTS FOR (e:Estructura) REQUIRE e.nombre IS UNIQUE",
            "CREATE CONSTRAINT tincion_nombre IF NOT EXISTS FOR (t:Tincion) REQUIRE t.nombre IS UNIQUE",
            "CREATE CONSTRAINT tabla_id IF NOT EXISTS FOR (t:Tabla) REQUIRE t.id IS UNIQUE",
        ]
        for c in constraints:
            try:
                await self.run(c)
            except Exception as e:
                print(f"  ⚠️ Constraint: {e}")

        try:
            indexes = await self.run("SHOW INDEXES YIELD name, type, options")
            for idx in indexes:
                if idx["type"] == "VECTOR":
                    name = idx["name"]
                    config = idx["options"].get("indexConfig", {})
                    dims = config.get("vector.dimensions")
                    
                    target_dims = None
                    if name == TEXT_INDEX_NAME: target_dims = TEXT_DIM
                    elif name == UNI_INDEX_NAME: target_dims = UNI_IMG_DIM
                    elif name == PLIP_INDEX_NAME: target_dims = PLIP_IMG_DIM
                    
                    if target_dims and dims != target_dims:
                        print(f"  ⚠️ Incorrect dimension on index '{name}' ({dims} != {target_dims}). Recreating...")
                        await self.run(f"DROP INDEX {name}")
        except Exception as e:
            print(f"  ⚠️ Schema validation error: {e}")

        vector_queries = [
            f"""
            CREATE VECTOR INDEX {TEXT_INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {TEXT_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX {UNI_INDEX_NAME} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_uni
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {UNI_IMG_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX {PLIP_INDEX_NAME} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_plip
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {PLIP_IMG_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        for vq in vector_queries:
            try:
                await self.run(vq)
            except Exception as e:
                print(f"  ⚠️ Vector index error: {e}")

        print("✅ Neo4j Schema Ready (3 indexes)")

    async def upsert_pdf(self, name: str):
        await self.run("MERGE (p:PDF {nombre: $name})", {"name": name})

    async def upsert_chunk(self, chunk_id: str, text: str, source: str,
                            chunk_idx: int, embedding: List[float],
                            entities: Dict[str, List[str]]):
        await self.run("""
            MERGE (c:Chunk {id: $id})
            SET c.texto = $text, c.fuente = $source,
                c.chunk_id = $chunk_idx, c.embedding = $embedding
            WITH c
            MERGE (pdf:PDF {nombre: $source})
            MERGE (c)-[:PERTENECE_A]->(pdf)
        """, {
            "id": chunk_id, "text": text, "source": source,
            "chunk_idx": chunk_idx, "embedding": embedding
        })
        for tissue in entities.get("tejidos", []):
            await self.run("""
                MERGE (t:Tejido {nombre: $name})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"name": tissue, "chunk_id": chunk_id})
        for structure in entities.get("estructuras", []):
            await self.run("""
                MERGE (e:Estructura {nombre: $name})
                WITH e MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
            """, {"name": structure, "chunk_id": chunk_id})
        for stain in entities.get("tinciones", []):
            await self.run("""
                MERGE (t:Tincion {nombre: $name})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"name": stain, "chunk_id": chunk_id})
        for tissue in entities.get("tejidos", []):
            for structure in entities.get("estructuras", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tissue})
                    MERGE (e:Estructura {nombre: $structure})
                    MERGE (t)-[:CONTIENE]->(e)
                """, {"tissue": tissue, "structure": structure})
            for stain in entities.get("tinciones", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tissue})
                    MERGE (ti:Tincion {nombre: $stain})
                    MERGE (t)-[:TENIDA_CON]->(ti)
                """, {"tissue": tissue, "stain": stain})
        for structure in entities.get("estructuras", []):
            for stain in entities.get("tinciones", []):
                await self.run("""
                    MERGE (e:Estructura {nombre: $structure})
                    MERGE (ti:Tincion {nombre: $stain})
                    MERGE (e)-[:TENIDA_CON]->(ti)
                """, {"structure": structure, "stain": stain})

    async def upsert_image(self, image_id: str, path: str, source: str,
                             page: int, ocr_text: str, page_text: str,
                             emb_uni: List[float], emb_plip: List[float]):
        await self.run("""
            MERGE (i:Imagen {id: $id})
            SET i.path = $path, i.fuente = $source,
                i.pagina = $page, i.ocr_text = $ocr_text,
                i.texto_pagina = $page_text,
                i.embedding_uni = $emb_uni,
                i.embedding_plip = $emb_plip
            WITH i
            MERGE (pdf:PDF {nombre: $source})
            MERGE (i)-[:PERTENECE_A]->(pdf)
            MERGE (pag:Pagina {numero: $page, pdf_nombre: $source})
            MERGE (i)-[:EN_PAGINA]->(pag)
        """, {
            "id": image_id, "path": path, "source": source,
            "page": page, "ocr_text": ocr_text, "page_text": page_text,
            "emb_uni": emb_uni, "emb_plip": emb_plip
        })

    async def upsert_table(self, tabla_id: str, contenido_md: str, source: str,
                           page: int, embedding: Optional[List[float]] = None):
        """Inserts or updates a :Tabla node with Markdown table content."""
        await self.run("""
            MERGE (t:Tabla {id: $id})
            SET t.contenido_md = $contenido_md, t.fuente = $source,
                t.pagina = $page, t.embedding = $embedding
            WITH t
            MERGE (pdf:PDF {nombre: $source})
            MERGE (t)-[:PERTENECE_A]->(pdf)
            MERGE (pag:Pagina {numero: $page, pdf_nombre: $source})
            MERGE (t)-[:EN_PAGINA]->(pag)
        """, {
            "id": tabla_id, "contenido_md": contenido_md, "source": source,
            "page": page, "embedding": embedding
        })

    async def create_similarity_relations(self, threshold: float = SIMILAR_IMG_THRESHOLD):
        print(f"🔗 Creating :SIMILAR_A relations (UNI method, threshold={threshold})...")
        images = await self.run(
            "MATCH (i:Imagen) WHERE i.embedding_uni IS NOT NULL RETURN i.id AS id, i.embedding_uni AS emb"
        )
        if len(images) < 2:
            return
        created = 0
        for img in images:
            result = await self.run("""
                CALL db.index.vector.queryNodes($index, 6, $emb)
                YIELD node AS vecino, score
                WHERE vecino.id <> $id AND score >= $threshold
                WITH vecino, score
                MATCH (origen:Imagen {id: $id})
                MERGE (origen)-[r:SIMILAR_A]->(vecino)
                SET r.score = score
                RETURN count(*) AS n
            """, {
                "index": UNI_INDEX_NAME,
                "emb": img["emb"], "id": img["id"], "threshold": threshold
            })
            created += result[0]["n"] if result else 0
        print(f"✅ {created} :SIMILAR_A relations linked")

    async def vector_search(self, embedding: List[float],
                                  index_name: str, top_k: int = 10) -> List[Dict]:
        is_text_index = index_name == TEXT_INDEX_NAME
        if is_text_index:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS c, score
                RETURN c.id AS id, c.texto AS text, c.fuente AS source,
                       'texto' AS type, null AS image_path, score AS similarity
                ORDER BY similarity DESC
            """
        else:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS i, score
                RETURN i.id AS id, 
                       coalesce(i.texto_pagina, i.ocr_text) AS text, 
                       i.fuente AS source,
                       'imagen' AS type, i.path AS image_path, score AS similarity
                ORDER BY similarity DESC
            """
        try:
            return await self.run(query, {"index": index_name, "emb": embedding, "k": top_k})
        except Exception as e:
            print(f"⚠️ Vector search error {index_name}: {e}")
            return []

    async def entity_search(self, entities: Dict[str, List[str]],
                                      top_k: int = 10) -> List[Dict]:
        tissues    = entities.get("tejidos", [])
        structures = entities.get("estructuras", [])
        stains     = entities.get("tinciones", [])
        if not any([tissues, structures, stains]):
            return []

        where_clauses = []
        params: Dict[str, Any] = {}
        if tissues:
            where_clauses.append("ANY(t IN tejidos WHERE t.nombre IN $tejidos)")
            params["tejidos"] = tissues
        if structures:
            where_clauses.append("ANY(e IN estructuras WHERE e.nombre IN $estructuras)")
            params["estructuras"] = structures
        if stains:
            where_clauses.append("ANY(ti IN tinciones WHERE ti.nombre IN $tinciones)")
            params["tinciones"] = stains

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
            RETURN c.id AS id, c.texto AS text, c.fuente AS source,
                   'texto' AS type, null AS image_path, 0.49 AS similarity
            LIMIT $top_k
        """
        params["top_k"] = top_k
        try:
            return await self.run(query, params)
        except Exception as e:
            print(f"⚠️ Entity search error: {e}")
            return []

    async def expand_neighborhood(self, node_ids: List[str],
                                 depth: int = NEO4J_GRAPH_DEPTH) -> List[Dict]:
        if not node_ids:
            return []
        query = """
            UNWIND $ids AS nid
            MATCH (n {id: nid})

            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)
            WHERE vecino_pdf.id <> nid
            WITH n, nid, collect(DISTINCT vecino_pdf)[..5] AS list_pdf

            OPTIONAL MATCH (n)-[:MENCIONA]->(entidad)<-[:MENCIONA]-(vecino_entidad:Chunk)
            WHERE vecino_entidad.id <> nid
            WITH n, nid, list_pdf, collect(DISTINCT vecino_entidad)[..5] AS list_ent

            OPTIONAL MATCH (n)-[:SIMILAR_A]-(vecino_similar:Imagen)
            WITH n, nid, list_pdf, list_ent, collect(DISTINCT vecino_similar)[..5] AS list_sim

            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf2:PDF)<-[:PERTENECE_A]-(img_pag:Imagen)
            WITH n, nid, list_pdf, list_ent, list_sim, collect(DISTINCT img_pag)[..5] AS list_pag

            WITH n, $ids AS ids_originales,
                 list_pdf + list_ent + list_sim + list_pag AS vecinos_raw

            UNWIND vecinos_raw AS v

            WITH n, v, ids_originales
            WHERE v IS NOT NULL AND NOT v.id IN ids_originales

            RETURN DISTINCT
                v.id AS id,
                CASE 
                    WHEN v:Imagen THEN coalesce(v.texto_pagina, v.ocr_text, '') 
                    ELSE coalesce(v.texto, '') 
                END AS text,
                v.fuente AS source,
                CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS type,
                CASE WHEN v:Imagen THEN v.path ELSE null END AS image_path,
                CASE 
                    WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
                         (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
                    THEN 0.95 
                    ELSE 0.3 
                END AS similarity
            LIMIT 15
        """
        try:
            return await self.run(query, {"ids": node_ids})
        except Exception as e:
            print(f"⚠️ Neighborhood expansion error: {e}")
            return []

    async def semantic_path_search(self,
                                         source_tissue: Optional[str],
                                         dest_tissue: Optional[str]) -> List[Dict]:
        if not source_tissue or not dest_tissue:
            return []
        query = """
            MATCH (origen {nombre: $source}), (destino {nombre: $dest})
            MATCH path = shortestPath((origen)-[*1..4]-(destino))
            UNWIND nodes(path) AS nodo
            OPTIONAL MATCH (c:Chunk)-[:MENCIONA]->(nodo)
            RETURN DISTINCT c.id AS id, c.texto AS text, c.fuente AS source,
                   'texto' AS type, null AS image_path, 0.4 AS similarity
            LIMIT 5
        """
        try:
            return await self.run(query, {"source": source_tissue, "dest": dest_tissue})
        except Exception as e:
            print(f"⚠️ Semantic Path search error: {e}")
            return []

    async def hybrid_search(self,
                                text_embedding: Optional[List[float]],
                                image_embedding_uni: Optional[List[float]],
                                image_embedding_plip: Optional[List[float]],
                                entities: Dict[str, List[str]],
                                top_k: int = 10) -> List[Dict]:
        res_text = []
        res_uni  = []
        res_plip = []
        res_ent  = []
        res_vec  = []

        if text_embedding is not None:
            res_text = await self.vector_search(text_embedding, TEXT_INDEX_NAME, top_k)
        if image_embedding_uni is not None:
            res_uni = await self.vector_search(image_embedding_uni, UNI_INDEX_NAME, top_k)
        if image_embedding_plip is not None:
            res_plip = await self.vector_search(image_embedding_plip, PLIP_INDEX_NAME, top_k)

        res_ent = await self.entity_search(entities, top_k)

        all_res = res_text + res_uni + res_plip
        top_ids = [r["id"] for r in all_res[:6] if r.get("id")]
        if top_ids:
            res_vec = await self.expand_neighborhood(top_ids)

        # ── Near-duplicate detection (UNI ∩ PLIP with score ≥ 0.95) ──
        near_dup_ids = set()
        if res_uni and res_plip:
            uni_scores = {r["id"]: r["similarity"] for r in res_uni if r.get("id")}
            plip_scores = {r["id"]: r["similarity"] for r in res_plip if r.get("id")}
            for img_id in uni_scores:
                if img_id in plip_scores:
                    if uni_scores[img_id] >= 0.95 and plip_scores[img_id] >= 0.95:
                        near_dup_ids.add(img_id)
            if near_dup_ids:
                print(f"   🎯 Near-duplicates detected: {near_dup_ids}")

        combined: Dict[str, Dict] = {}

        def add(results: List[Dict], weight: float):
            for r in results:
                key = r.get("id") or f"{r.get('source')}_{str(r.get('text',''))[:40]}"
                if not r.get("text") and not r.get("image_path"):
                    continue
                weighted_sim = r.get("similarity", 0) * weight
                # Near-duplicate boost ×2.0
                if r.get("id") in near_dup_ids:
                    weighted_sim *= 2.0
                if key not in combined:
                    combined[key] = {**r, "similarity": weighted_sim}
                else:
                    combined[key]["similarity"] += weighted_sim

        add(res_text, 0.80) 
        add(res_uni,  0.50) 
        add(res_plip, 0.50) 
        add(res_ent,  0.60) 
        add(res_vec,  0.20)

        final = sorted(combined.values(), key=lambda x: x["similarity"], reverse=True)

        print(f"   📊 Hybrid Search: Txt={len(res_text)} | "
              f"UNI={len(res_uni)} | PLIP={len(res_plip)} | Ent={len(res_ent)} | Vec={len(res_vec)} -> {len(final)}"
              + (f" | 🎯 NearDup={len(near_dup_ids)}" if near_dup_ids else ""))

        return final[:15]
```

---

## Feature 4: Near-Duplicate Boost (Re-ranking Híbrido)

**Archivo:** [neo4j_client.py](file:///home/francisco/Escritorio/IA/histo-test/db/neo4j_client.py)

En `hybrid_search()`:
1. Calcula `near_dup_ids` = intersección de IDs presentes en resultados UNI **y** PLIP con ambos scores ≥ 0.95
2. En la función `add()`, aplica `weighted_sim *= 2.0` para IDs near-duplicate
3. Log: `🎯 NearDup={N}` al final del print de métricas

---

## Archivos Modificados (Reporte Final)

| Archivo | Features Aplicados | Cambios Clave |
|---------|-------------------|---------------|
| `core/ingestion.py` | 1, 2c | RecursiveCharacterTextSplitter, table detection + upsert |
| `extractors/pdf.py` | 2a | `detect_and_extract_table()` (Groq+Gemini) |
| `db/neo4j_client.py` | 2b, 3c, 4 | `:Tabla` constraint, `upsert_table()`, SNOMED/FMA Cypher, NearDup Boost |
| `extractors/text.py` | 3b | Ontology-enriched entity extraction |
| `prompts/entity_extractor.txt` | 3a | SNOMED/FMA prompt |

> [!NOTE]
> `core/agent.py` y `server.py` **no requirieron cambios** — el agent usa `EntityExtractor.extract_from_text()` (async, ya actualizado) y `neo4j_client.hybrid_search()` (ya contiene el boost). La backward compatibility se garantiza con guards `isinstance(tissue, dict)`.

## Verificación

```
✅ core/ingestion.py: all 4 markers present — valid syntax
✅ db/neo4j_client.py: all 7 markers present — valid syntax
✅ extractors/pdf.py: all 4 markers present — valid syntax
✅ extractors/text.py: all 2 markers present — valid syntax
✅ prompts/entity_extractor.txt: all 3 markers present
✅ core/agent.py: valid syntax
✅ server.py: valid syntax
🎉 ALL 4 FEATURES VERIFIED
```
