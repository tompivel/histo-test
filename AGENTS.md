# AGENTS.md -- Baseline Repository Guide

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python >=3.10 (primary), vanilla JavaScript (frontend) |
| **Package Manager** | `uv` (Python), `npm` (scripts only, no JS deps) |
| **Web Framework** | FastAPI + Uvicorn |
| **LLM Orchestration** | LangGraph (StateGraph), LangChain Core |
| **Primary LLM** | Groq (meta-llama/llama-4-scout-17b-16e-instruct) |
| **Fallback LLM Vision** | Google Gemini (`langchain-google-genai`) |
| **Graph Database** | Neo4j AuraDB (async driver, 3 vector indexes: 384d/512d/1024d cosine) |
| **Vector Memory** | Qdrant (`qdrant-client`, in-memory `:memory:` mode, persistent to `qdrant_memoria/`) |
| **Medical Vision** | UNI (`timm`, MahmoodLab, 1024d), PLIP (`vinid/plip`, 512d), CONCH (`mahmoodlab/CONCH`) |
| **Text Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384d) |
| **PDF / OCR** | PyMuPDF, pdf2image, pytesseract |
| **Deep Learning** | PyTorch >=2.0, NumPy, Pillow |
| **Frontend** | Vanilla HTML/CSS/JS (no framework), Inter font |
| **Prompts** | Externalized in `prompts/` directory (8 `.txt` files) |
| **Config / Secrets** | `.env` file, hybrid Colab/local detection in `utils/config.py` |

## Build / Run / Test Commands

| Command | Action |
|---|---|
| `npm run dev` | Start dev server with hot-reload (uvicorn) |
| `npm run start` | Start production server |
| `uv run python server.py` | Start FastAPI server directly |
| `uv run python main.py` | Run CLI interactive mode |
| `uv run python run_etl.py` | Run ETL pipeline (wipe + re-ingest PDFs into Neo4j) |
| `uv sync` | Install/ sync Python dependencies from `uv.lock` |
| **No test framework or linter configured** | -- |

## Project Structure

```
histo-test/
├── main.py                  # CLI entrypoint -> core.cli.interactive_mode()
├── server.py                # FastAPI server (serves client/ static frontend)
├── run_etl.py               # Standalone ETL runner
├── pyproject.toml           # Python project config + dependencies
├── package.json             # npm scripts (dev, start)
├── uv.lock                  # uv dependency lockfile
│
├── core/                    # Core business logic
│   ├── agent.py             # Neo4jHistologyAgent (LangGraph orchestrator, 6-node workflow)
│   ├── cli.py               # Interactive CLI mode
│   └── ingestion.py         # IngestionPipeline (PDF -> Chunks/Images/Tables -> Neo4j)
│
├── db/                      # Database clients
│   ├── neo4j_client.py      # Neo4jClient (async, CRUD, hybrid search, graph traversal)
│   └── memory.py            # SemanticMemory (Qdrant vector conversation store)
│
├── models/                  # ML model wrappers
│   ├── classifiers.py       # SemanticClassifier (domain guardrail)
│   └── vision.py            # PlipWrapper + UniWrapper (medical image embeddings)
│
├── extractors/              # Data extraction modules
│   ├── pdf.py               # PDFImageExtractor (PDF parsing, OCR, table detection)
│   └── text.py              # EntityExtractor + TopicExtractor (NLP entity extraction)
│
├── utils/                   # Utilities
│   ├── config.py            # Global config, secrets, LangSmith setup
│   ├── prompt_loader.py     # load_prompt() from prompts/ dir
│   └── resilience.py        # Retry logic for LLM/embedding API calls
│
├── client/                  # Static frontend (vanilla HTML/CSS/JS)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── prompts/                 # 8 externalized LLM prompt templates (.txt)
├── pdf/                     # Source PDF manuals for ETL ingestion
├── notebooks/               # Jupyter/Colab notebooks (RAG_Ingestion_Chat.ipynb)
├── docs/                    # Manual testing docs, architecture analysis
├── history/                 # Migration/refactoring summaries
├── monolithic/              # Historical backup of old monolithic versions
├── imagenes_extraidas/      # Extracted PDF images (gitignored)
├── imagenes_chat/           # User-uploaded chat images
└── qdrant_memoria/          # Persistent Qdrant storage on disk
```

## Entry Points

- **Web UI**: `server.py` -- FastAPI app serving `client/index.html` as the A2UI chat interface.
- **CLI**: `main.py` -> `core/cli.py` -- Terminal-based chat with image upload support.
- **ETL**: `run_etl.py` -> `core/ingestion.py` -- Designed for Colab GPU; populates Neo4j from PDFs in `pdf/`.

## Key Architecture Notes

- The `Neo4jHistologyAgent` in `core/agent.py` is a 6-node LangGraph StateGraph: verify_domain -> visual_analysis (optional) -> retrieve_memory -> retrieve_neo4j -> generate_response / out_of_domain.
- Neo4j schema: node labels `PDF`, `Chunk`, `Imagen`, `Tejido`, `Estructura`, `Tincion`, `Pagina`, `Tabla` with relationships `PERTENECE_A`, `MENCIONA`, `CONTIENE`, `TENIDA_CON`, `SIMILAR_A`, `EN_PAGINA`.
- Three vector indexes: `histo_text` (384d), `histo_img_uni` (1024d), `histo_img_plip` (512d).
- No Docker infrastructure exists.

## Strict Conventions
- Do not run `run_etl.py` locally. Instead, if it is necessary to run, specify the user to do it. Generally, this step will be performed externally on a Google Collab notebook.
- All variables and internal code overall has to be written in english. Spanish kept for the end user and ontologies used in Neo4J schema. 
