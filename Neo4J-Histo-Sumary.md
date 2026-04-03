# Neo4J-Histo Multimodal RAG Agent: Technical Summary

## 1. Overview
`ne4j-histo.py` is an advanced multimodal Retrieval-Augmented Generation (RAG) system tailored specifically for histological data. It combines visual models with text interpretation to help users query histological textbooks, identify structures from uploaded images, and differentiate between similar cells or tissues.

The system builds its retrieval foundation on **Neo4j** (Graph Database), utilizing both classic graph traversals and multiple high-dimensional vector index searches to pinpoint relevant textbook sections and reference images.

---

## 2. Agent Structure & Workflow (`LangGraph`)

The RAG logic is orchestrated as an `AgentState` state machine via **LangGraph**. The workflow comprises several specific nodes connected directionally:

1. **`inicializar`**: Prepares the LangGraph state, initiates memory context, tracks start time, and manages query parameters.
2. **`procesar_imagen`**: Analyzes user-provided images. It detects if there's a new image or actively retains an image from previous conversation turns. It invokes multimodal models to embed the image and requests an LLM-based summary of its visual attributes.
3. **`clasificar`**: Uses the `ClasificadorSemantico` to score if the user's inquiry actually belongs to the histology domain. If it's unrelated, the flow short-circuits to the `fuera_temario` node and exits early.
4. **`generar_consulta`**: Optimizes the user's verbose text prompt into succinct query keywords strictly for better semantic search performance. 
5. **`buscar_neo4j`**: An intricate Hybrid Search within Neo4j. It fetches results based on Text Embeddings, UNI Image Embeddings, PLIP Embeddings, and Graph Entity Relationships (like related "tejido" or "estructura"). 
6. **`filtrar_contexto`**: Drops results below a configurable similarity score (`0.45`), reducing token bloat and maintaining strict relevance. 
7. **`analisis_comparativo`**: A specialized, highly-critical node that forces the LLM to rigidly contrast the user's image against the top retrieved manual reference images to ensure they depict the same biological structure.
8. **`generar_respuesta`**: Formulates the final output based exclusively on what exists in the vector space context. It enforces citations to manual pages or manual images, refusing to provide "hallucinated" general medical advice.
9. **`finalizar`**: Saves the end-to-end conversation into a persistent Qdrant memory buffer and concludes execution.

---

## 3. Key Classes and Functions

* **`AsistenteHistologiaNeo4j`**: The core application module. Initializes connections to APIs, instantiates Large Language Models (LLM) and Embedding models, handles database setups, defines the layout of the LangGraph, and serves the `consultar` execution method.
* **`ExtractorImagenesPDF`**: Scrapes histological textbook PDFs. It employs `fitz` (PyMuPDF) to extract pure embedded image objects or defaults to full-page rendering when direct images are too small. It simultaneously applies OCR (`pytesseract`) to capture contiguous captions.
* **`Neo4jClient`**: Provides a robust asynchronous wrapper around the Neo4j database driver. Because Neo4j acts simultaneously as a graph database and a multi-dimensional vector store, this class orchestrates many distinct layers of logic:
  * **`crear_esquema()`**: Initializes the database constraints (like unique IDs) and Cypher vector indexes. It validates dimension boundaries for the three indices (`INDEX_TEXTO` [384], `INDEX_UNI` [1024], and `INDEX_PLIP` [512]) and rebuilds them if structurally mismatched.
  * **`upsert_chunk()` & `upsert_imagen()`**: Creates the nodes and manually wires up interconnected graph paths. For instance, when it saves a chunk of text, it splits the identified tissues, structures, and stains via `MERGE` relationships.
    ```cypher
    // Example topology connection inside upsert_chunk()
    MERGE (t:Tejido {nombre: $nombre})
    WITH t MATCH (c:Chunk {id: $chunk_id})
    MERGE (c)-[:MENCIONA]->(t)
    ```
  * **`crear_relaciones_similitud()`**: Pre-computes visual similarity cache. It performs vector searches of image embeddings against other images already within the database, and creates permanent `[SIMILAR_A]` relationships between them if they score highly. This heavily accelerates future query speeds.
  * **Graph Traversal Methods (`busqueda_por_entidades`, `expandir_vecindad`, `busqueda_camino_semantico`)**: Rather than vector algorithms, these use native graph topology logic. `expandir_vecindad` grabs "1-hop neighbors"—if the system finds a great text chunk or image, it queries for contextually adjacent nodes (same PDF page, shared biological entity, or pre-computed `SIMILAR_A` links).
    * **Synthetic similarity scoring**: Because graph traversal produces no real cosine score, `expandir_vecindad` assigns hardcoded proxy values. Neighbors on the **same page or same source PDF** as the seed node get `0.95`; all other graph neighbors get `0.30`.
      ```cypher
      CASE 
        WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
             (n:Chunk  AND v:Imagen AND n.fuente = v.fuente) THEN 0.95 
        ELSE 0.30
      END AS similitud
      ```
    * **⚠️ Design questions worth exploring**: The gap between `0.95` and `0.30` is very steep. A neighbor chunk that shares a `Tejido` entity with the seed (i.e., they explicitly discuss the same structure) gets the same flat `0.30` as one linked by a very loose graph path—despite being semantically more relevant. Some improvements to consider:
        * Use a mid-range value (e.g., `0.60`) for entity-shared neighbors vs. `0.30` for page-siblings from a different PDF.
        * Compute actual cosine similarity between the seed embedding and the neighbor's embedding at query time, replacing the heuristic entirely.
        * Distinguish between the four types of expansion paths (`list_pdf`, `list_ent`, `list_sim`, `list_pag`) with their own individual scores rather than collapsing them into a binary.
  * **`busqueda_hibrida()`**: The overarching search orchestrator. It performs isolated lookups against the text vector, UNI vector, PLIP vector, explicit entities, and neighboring paths, and applies a rigid weighting formula to surface the best global matches.
    ```python
    # Final aggregation logic inside busqueda_hibrida()
    agregar(res_texto, 0.80) # Text carries the highest weight
    agregar(res_uni,   0.50) # UNI visual matching
    agregar(res_plip,  0.50) # PLIP visual matching
    agregar(res_ent,   0.60) # Graph Entity matching (Crucial for specific organs!)
    agregar(res_vec,   0.20) # Extracted neighborhood graph context
    ```
* **`setup_langsmith_environment` & Python's `*args`/`**kwargs`**: 
  * This function attempts to safely initialize **LangSmith** (an observability and debugging platform for LLMs). It maps your API keys into `os.environ` and tries to connect to the LangChain tracing server.
  * **Graceful Degradation**: If LangSmith isn't installed or fails to initialize, the script catches the exception and provides a mock `dummy_traceable` wrapper. This is a failsafe that prevents the entire application from crashing; it ensures that any `@traceable` decorators used later in the code simply pass through and do nothing.
  * **What are `*args` and `**kwargs`?** `*args` captures positional arguments as a tuple; `**kwargs` captures keyword arguments as a dictionary. In `dummy_traceable`, they are a "catch-all" so the function absorbs any parameters without throwing a `TypeError`.
  * **Decorator pattern & `callable()`**: A decorator in Python is a function that takes a function as argument and returns one. `def decorator(func): return func` is the canonical form — a no-op that returns the original function unchanged. The `callable(args[0])` check handles two different `@traceable` usage styles: when used bare (`@traceable`, Python passes the function directly as `args[0]`, which is callable) versus with arguments (`@traceable(run_type="llm")`, which passes a string, not callable, so the wrapper must return a decorator instead). `callable()` is a Python built-in that returns `True` if an object can be invoked as a function.
* **`invoke_con_reintento` & `embed_query_con_reintento` (Resilience Wrappers)**: These wrappers are designed to gracefully handle API throttling. 
  * **Why the retries?** Because the heavy Language Model reasoning is performed via the external **Groq cloud API** (which enforces strict usage rate limits), the `invoke_con_reintento` wrapper catches `429 Resource Exhausted` errors and applies an incremental delay (a backoff timer) to retry automatically instead of crashing the application during heavy indexation workloads.
  * **Are embeddings generated via an external server?** You might notice `embed_query_con_reintento` also attempting to handle `429` API errors, implying text embeddings are processed on a server. However, this is actually **vestigial legacy code** from a prior version of the script that utilized Google Gemini's cloud API for text embeddings. If you check `_init_modelos()`, the author swapped the text embeddings to use `sentence-transformers/all-MiniLM-L6-v2` locally. In the current implementation, all text and image embeddings (`sentence-transformers`, `UNI`, `PLIP`) run **100% locally on your hardware**, meaning the `embed_*` retry logic will never actually be triggered.
* **`SemanticMemory`**: Orchestrates stateful behavior across user chat sessions. This class uses a secondary lightweight vector database (`Qdrant`) to track and persist summary embeddings of the conversation history.
* **`ClasificadorSemantico`**: Determines whether a conversational prompt is about histology using LLM arbitration combined with cosine-similarity comparison against predefined domain-specific keyword anchors. 
* **`PlipWrapper` & `UniWrapper`**: Lightweight wrapper classes built to abstract the complexities of loading and executing the heavy vision models.
  * **`load()`**: Automatically fetches the pretrained model weights and architectures from the Hugging Face Hub. It loads the `vinid/plip` model using the native `transformers` library, and the `MahmoodLab/UNI` model using the `timm` (PyTorch Image Models) library. It maps the models directly to the available hardware (`self.device`, CPU or GPU) and prepares the mandatory image transformations (`CLIPProcessor` for PLIP and TIMM-based transforms for UNI), switching the models into `.eval()` mode.
  * **`embed_image(image_path)`**: The core function that actually looks at a picture and generates an embedding. It opens the physical image file, applies the aforementioned transformations to standardize pixel values, and pushes the image tensor through the neural network. To ensure maximum efficiency and avoid memory leaks, the forward pass is placed strictly inside `torch.inference_mode()`. Once processing is completed, it scoops out the final pooled vector block (a `512` array for PLIP, and a `1024` array for UNI) and collapses it into a flat numpy array ready to be ingested into the Neo4j vector index.

---

## 4. Embeddings & Models in Use

### **Embeddings**
To support multimodal semantics, the system leverages three parallel vector environments:
1. **Text**: Handled by **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** (384 dimensions). Responsible for establishing semantic bridges between chunks of textbook paragraphs and user prompts.
2. **PLIP (Pathology Language-Image Pretraining)**: Sourced from `vinid/plip`, it is a fine-tuned version of CLIP specifically designed for histopathology (512 dimensions).
3. **UNI**: Sourced from `hf_hub:MahmoodLab/UNI`, a monumental ViT-L/16 model explicitly trained for high-resolution computational pathology inference (1024 dimensions).

### **Large Language Model (LLM)**
* **`meta-llama/llama-4-scout-17b-16e-instruct`**: The logical brain of the agent, powered over external APIs strictly utilizing the **Groq** cloud engine to ensure immensely fast token generation speeds.

---

## 4.1. Multispace Embeddings and Similarity Search Architecture

### How Different Embedding Dimensions Coexist
A common question in multimodal vector databases is how vectors of completely different sizes (e.g., 384, 512, and 1024 dimensions) can exist in the same database and be searched effectively. 

The RAG system does **not** concatenate or force these vectors into a single generic space. Instead, Neo4j allows the creation of **completely independent Vector Indexes** bound to specific dimensions. Each type of embedding gets its own strict index, separating chunks of text from the two distinct types of image embeddings:
```python
# From Neo4jClient.crear_esquema()
# 1. INDEX_TEXTO (384 dimensions)
f"""CREATE VECTOR INDEX {INDEX_TEXTO} IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {{indexConfig: {{`vector.dimensions`: {DIM_TEXTO}, `vector.similarity_function`: 'cosine'}} }}"""

# 2. INDEX_UNI (1024 dimensions)
f"""CREATE VECTOR INDEX {INDEX_UNI} IF NOT EXISTS
FOR (i:Imagen) ON i.embedding_uni
OPTIONS {{indexConfig: {{`vector.dimensions`: {DIM_IMG_UNI}, `vector.similarity_function`: 'cosine'}} }}"""
```

### Performing Similarity Search Across Spaces
When a user inputs a query (with both text and an image), the system generates independent embeddings for the text and the discrete image formats (UNI and PLIP). 

Cosine similarity is **never calculated between a text vector and an image vector**. Instead, the system performs isolated searches within each specific vector index, scoring nodes (0 to 1) purely against their own dimension space. 
```python
# From Neo4jClient.busqueda_hibrida()
# Independent separate searches against isolated indices
if texto_embedding:
    res_texto = await self.busqueda_vectorial(texto_embedding, INDEX_TEXTO, top_k)
if imagen_embedding_uni:
    res_uni = await self.busqueda_vectorial(imagen_embedding_uni, INDEX_UNI, top_k)
```

After retrieving the top matches from each isolated query, the application aggregates and merges the results using a **weighted scoring approach** (e.g., `res_texto` carries an `0.80` weight, while visual results like `res_uni` carry a `0.50` weight).

### What are UNI and PLIP and their roles in the code?
The system actually uses **BOTH** UNI and PLIP concurrently for image embeddings, leveraging their distinct strengths.

1. **UNI (Universal Model for Computational Pathology)**: An immense Vision Transformer (ViT-L/16) created by MahmoodLab, heavily pre-trained on millions of histological whole-slide images. It acts as the primary "visual brain" for finding identical or structurally similar tissues, and its raw output is a 1024-dimension vector.
2. **PLIP (Pathology Language-Image Pretraining)**: Sourced from `vinid/plip`, it is a fine-tuned version of CLIP specifically designed for histopathology. Its raw output is a 512-dimension vector.

In this application, when a visual query is processed or when images are being indexed, the system passes the image through **both** models to extract 1024 visual features from UNI and 512 visual features from PLIP. The database aggregates similarities from both models during hybrid search to get a stronger combined result irrespective of text keywords.
```python
class UniWrapper:
    # ...
    def embed_image(self, image_path: str) -> np.ndarray:
        # Extracts 1024 visual features
        # ...
        
class PlipWrapper:
    # ...
    def embed_image(self, image_path: str) -> np.ndarray:
        # Extracts 512 visual features
        # ...

# During hybrid search, both are used together:
if imagen_embedding_uni:
    res_uni = await self.busqueda_vectorial(imagen_embedding_uni, INDEX_UNI, top_k)
if imagen_embedding_plip:
    res_plip = await self.busqueda_vectorial(imagen_embedding_plip, INDEX_PLIP, top_k)
```

---
## 5. Laptop execution without a GPU (Google Colab consideration)

### **Can this application run locally on a laptop without a GPU?**
**Yes, it can run entirely on a CPU, but with significant performance caveats.**

#### What runs locally?
The application initializes the `device` property by executing `"cuda" if torch.cuda.is_available() else "cpu"`. If no NVIDIA GPU is detected, PyTorch will gracefully step down to utilizing the CPU.
* The **LLM component** (Llama-4) is queried remotely through Groq's high-speed cloud APIs, saving your computer from the impossible memory burden of hosting a 17B model. 
* *However*, the **Embedding and Vision components** (UNI, PLIP, and Sentence-transformers) **run 100% locally**. 

#### Performance Impact on CPU:
* **Indexation Phase (Very Slow)**: If you possess textbooks with large amounts of pages and images, running OCR alongside the UNI ViT-L vision transformer purely on a laptop's CPU to vector-index the data will take an exorbitant amount of time (potentially hours per PDF). 
* **Query Phase (Acceptable)**: Once textbooks are completely embedded/indexed into the Neo4j database, querying the system is comparatively light. Passing a single user image to UNI/PLIP during query generation will pause execution for roughly ~4–10 seconds on a modern CPU, which is manageable.

#### **Recommendation for Google Colab fallback:**
If you need to re-index large textbooks effectively, falling back to a **Google Colab environment (using a T4 or L4 GPU)** is highly recommended. The GPU acceleration will cut processing, embeddings compilation, and OCR routines from hours down to just a few minutes. If you want to remain on a laptop, consider letting the indexation job execute overnight.
