import os
from dotenv import load_dotenv

# =============================================================================
# HYBRID SECRETS MANAGEMENT (LOCAL vs COLAB)
# =============================================================================
try:
    from google.colab import userdata
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False
    class userdata:
        """
        Convenience class mocking google.colab.userdata behavior.
        In local environments, it intercepts calls and pulls from os.environ (.env).
        """
        @staticmethod
        def get(key):
            return os.environ.get(key)

# Load environment variables from .env (Only affects Local)
load_dotenv()

# Verify HF_TOKEN
HF_TOKEN = userdata.get("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("✅ Logged into Hugging Face")
    except Exception as e:
        print(f"⚠️ HF login error: {e}")
else:
    print("⚠️ HF_TOKEN not found (required to download UNI/PLIP)")


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SIMILARITY_THRESHOLD  = 0.45
"""Base similarity threshold. Matches with lower scores may be ignored by basic searches."""

# Embedding dimensions
TEXT_DIM        = 384
"""Dimensionality of the latent array for classic manual text. Default: all-MiniLM-L6-v2."""
UNI_IMG_DIM      = 1024
"""Dimensionality of the visual embedding produced by the pre-trained UNI model."""
PLIP_IMG_DIM     = 512
"""Dimensionality of the feature vector projected by the lightweight PLIP model."""

IMG_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imagenes_extraidas")
"""Absolute path to the base directory where extracted images from the manuals will be dumped or read."""

PDFS_DIR       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pdf")
"""Default directory where PDF manuals reside for batch ingestion feeding the RAG."""

# Neo4j Indexes
TEXT_INDEX_NAME = "histo_text"      # Text Vector
"""Name of the vector index registered for textual Chunks in Neo4j config."""
UNI_INDEX_NAME   = "histo_img_uni"   # UNI Image
"""Identifier of the VectorIndex reserved for abstract Node UNI sub-embeddings."""
PLIP_INDEX_NAME  = "histo_img_plip"  # PLIP Image
"""Explicit identifier of the vector index associated with visual PLIP tensors."""

NEO4J_GRAPH_DEPTH     = 2
"""Maximum jump radius/depth for indirect Cyber algorithmic neighborhood expansions."""

SIMILAR_IMG_THRESHOLD = 0.85 
"""Safety margin or strict threshold to create automatic :SIMILAR_A relations in Neo4j."""

DISCRIMINATORY_FEATURES = [
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
"""Static base heuristics injected into the LLM logic to orient pathologic reasoning."""

# Semantic anchors for domain classification
SEMANTIC_ANCHORS = [
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
]
"""Set of semantic anchor points (Vectors) defining the referential baseline for SemanticClassifier."""

def setup_langsmith_environment():
    """
    Initializes unified tracing by injecting static configuration flags.
    
    It intercepts all LangChain flows executed during the workflow and feeds them to the LangSmith UI.

    Returns:
        tuple: (status, traceable_decorator, client)
    """
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
        print(f"✅ LangSmith — Project: {os.environ.get('LANGCHAIN_PROJECT')}")
        return True, traceable, client
    except Exception as e:
        print(f"⚠️ LangSmith unavailable: {e}")
        def dummy_traceable(*args, **kwargs):
            def decorator(func): return func
            if len(args) == 1 and callable(args[0]): return args[0]
            return decorator
        return False, dummy_traceable, None

# Export LangSmith settings
LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith_environment()

def _safe(value, default: str = "") -> str:
    """
    Generic sanitization of null strings coming from irregular ORM property extractions.
    """
    return value if isinstance(value, str) and value else default
