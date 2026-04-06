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
