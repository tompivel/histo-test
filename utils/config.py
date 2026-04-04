import os
from dotenv import load_dotenv

# =============================================================================
# GESTIÓN HÍBRIDA DE SECRETOS (LOCAL vs COLAB)
# =============================================================================
try:
    from google.colab import userdata
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False
    class userdata:
        """
        Clase de conveniencia emulando el comportamiento de google.colab.userdata.
        En entornos locales, intercepta las llamadas y extrae de os.environ (.env).
        """
        @staticmethod
        def get(key, default=None):
            return os.environ.get(key, default)

# Cargar variables de entorno desde .env (Solo afecta en Local)
load_dotenv()

# Verificar HF_TOKEN
HF_TOKEN = userdata.get("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("✅ Logueado en Hugging Face")
    except Exception as e:
        print(f"⚠️ Error login HF: {e}")
else:
    print("⚠️ HF_TOKEN no encontrado (necesario para descargar UNI/PLIP)")

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
SIMILARITY_THRESHOLD  = 0.45
"""Umbral de similitud base. Coincidencias con puntajes por debajo pueden ser ignoradas por búsquedas básicas."""

# Dimensiones de embeddings
DIM_TEXTO        = 384
"""Dimensiones del array latente para el texto clásico de manual. Default: all-MiniLM-L6-v2."""
DIM_IMG_UNI      = 1024
"""Dimensionalidad del embedding visual producido por el modelo pre-entrenado UNI."""
DIM_IMG_PLIP     = 512
"""Dimensionalidad del feature vector proyectada por el modelo ligero PLIP."""

# Adjust path to account for being in utils/
DIRECTORIO_IMAGENES   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imagenes_extraidas")
"""Ruta absoluta del directorio base donde se volcarán o se leerán las imágenes extraídas de los manuales."""

DIRECTORIO_PDFS       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pdf")
"""Directorio por defecto donde residen los manuales PDF para alimentación o ingestion en lote del RAG."""

# Índices Neo4j
INDEX_TEXTO = "histo_text"      # Text Vector
"""Nombre del índice vector registrado para Chunks textuales en la configuración de Neo4j."""
INDEX_UNI   = "histo_img_uni"   # UNI Image
"""Identificador del VectorIndex reservado para sub-embeddings UNI sobre Nodos abstractos (Imagénes)."""
INDEX_PLIP  = "histo_img_plip"  # PLIP Image
"""Identificador explícito del índice vectorial asociado a tensores PLIP visuales."""

NEO4J_GRAPH_DEPTH     = 2
"""Profundidad o radio de saltos máximo para operaciones indirectas en Cypher de algoritmos de vecindario."""

SIMILAR_IMG_THRESHOLD = 0.85 
"""Margen de seguridad o umbral estricto para crear en Neo4j relaciones automáticas :SIMILAR_A durante las iteraciones."""

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
"""Heurísticas estáticas base inyectadas al LLM en prompts predefinidos a fines de orientar razonamientos patológicos."""

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
]
"""Conjunto de puntos (Vectores) anclas semánticos que definen la base referencial del ClasificadorSemantico per se."""

def setup_langsmith_environment():
    """
    Inicializa el rastreo unificado (Tracing) inyectando flags de configuración estáticas.
    
    Toma provecho de la llave LANGSMITH_API_KEY para interceptar todos los flujos LangChain
    ejecutados en cascada durante el workflow y pasarlos al front-end gráfico de Smith Langchain.

    Returns:
        tuple:
            - estado_activacion (bool): Si la inicialización fue validada satisfactoriamente.
            - traceable: Decorador de función oficial langsmith.traceable listo (o el fallback neutro o estéril).
            - langsmith_client: El cliente o None si ha abortado localmente.
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
        print(f"✅ LangSmith — Proyecto: {os.environ.get('LANGCHAIN_PROJECT')}")
        return True, traceable, client
    except Exception as e:
        print(f"⚠️ LangSmith no disponible: {e}")
        def dummy_traceable(*args, **kwargs):
            def decorator(func): return func
            if len(args) == 1 and callable(args[0]): return args[0]
            return decorator
        return False, dummy_traceable, None

# Export LangSmith settings
LANGSMITH_ENABLED, traceable, langsmith_client = setup_langsmith_environment()

def _safe(value, default: str = "") -> str:
    """
    Sanitización genérica de strings nulos proviniendo de extracción irregular de propiedades ORM.

    Args:
        value (any): Propiedad a verificar.
        default (str, opcional): Cadena a sustituir si `value` está manchado (None o vacío). Usa vacío por defecto.

    Returns:
        str: El string procesado y validado de manera segura para consumo de texto.
    """
    return value if isinstance(value, str) and value else default
