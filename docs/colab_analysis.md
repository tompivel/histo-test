# Análisis de Ingestión en Google Colab y Embeddings

## 1. Viabilidad de separación de concerns: Ingestión vs Inferencia (RAG)

**¿El pre-procesamiento y la generación de embeddings se ejecutan cada vez que se usa el [modo_interactivo](file:///home/tom/histo-test/ne4j-histo_bak.py#2207-2303)?**
¡No! Actualmente el [modo_interactivo](file:///home/tom/histo-test/ne4j-histo_bak.py#2207-2303) está puramente diseñado para inferencia. Llama a `asistente.initialize()`, lo cual se conecta a la base de datos Neo4j e inicializa la memoria local Qdrant, pero **no** ejecuta OCR ni procesa PDFs. Ya hemos separado conceptualmente la ingestión del consumo RAG.

**¿Es viable ejecutar el pre-procesamiento en un Google Colab Notebook?**
Es **altamente recomendable** e ideal por las siguientes razones:
1. **Poder de Cómputo Gratuito (GPU):** La extracción de imágenes multimodales con los modelos médicos UNI y PLIP requiere cargar de grandes parámetros en VRAM (ViT-L/16 requiere mucha VRAM). Colab ofrece aceleradores potentes gratuitamente (como la T4).
2. **Separación de responsabilidades (Decoupling):** El notebook de Colab actuará únicamente como ETL (Extract, Transform, Load). Extrae los PDF, genera los embeddings masivos, y hace un [upsert](file:///home/tom/histo-test/db/neo4j_client.py#147-155) enviándolos remotamente a la base de datos de Neo4j en la nube (ej. Neo4j Aura). 
3. **App Liviana:** Al delegar esto a Colab, la máquina o terminal local ejecuta simplemente [core/cli.py](file:///home/tom/histo-test/core/cli.py), por lo cual, no se traba ni requiere cargar bibliotecas gigantes y lentas de OCR (`pytesseract`, `pdf2image`) para iniciar chateos.

**¿Dónde se integraba y bajo qué condiciones (antes del refactor)?**
En el archivo original monolítico ([ne4j-histo_bak.py](file:///home/tom/histo-test/ne4j-histo_bak.py)), la ingestión de documentos sucedía al inicio de la función [modo_interactivo()](file:///home/tom/histo-test/ne4j-histo_bak.py#2207-2303). Las rutinas [procesar_contenido_base()](file:///home/tom/histo-test/ne4j-histo_bak.py#2036-2044) y [extraer_y_preparar_temario()](file:///home/tom/histo-test/ne4j-histo_bak.py#2045-2055) se ejecutaban al principio de **cada sesión**. Sin embargo, la indexación real intensiva en sí ([indexar_en_neo4j()](file:///home/tom/histo-test/ne4j-histo_bak.py#2056-2137), que se encarga de subir los chunks a la BD con los embeddings generados) solo se desencadenaba si se le pasaba el argumento explicitamente al llamar el script por consola (por ejemplo, ejecutando `python ne4j-histo_bak.py --reindex` o `--force`).

Con el refactor, esto está totalmente desacoplado y centralizado a través del módulo [PipelineIngestion](file:///home/tom/histo-test/core/ingestion.py#14-137) ([core/ingestion.py](file:///home/tom/histo-test/core/ingestion.py)).

---

## 2. Flujo de Vectores y Embeddings (Consultas e Imágenes)

**¿Dónde se integran las queries del usuario? ¿Van una o dos veces?**
Exactamente, la pregunta (`input`) se somete a embedding **dos** veces por turnos diferentes:
1. **Filtro Guardrail ([ClasificadorSemantico](file:///home/tom/histo-test/ne4j-histo_bak.py#917-1034)):** Para medir la similitud puramente geométrica del string de entrada contra el espacio latente de las "anclas semánticas" del manual de histología.
2. **Sistema de Retirada ([nodo_recuperar_neo4j](file:///home/tom/histo-test/core/agent.py#209-238)):** Para mandarlo a la búsqueda de similitud coseno vectorial pura en Neo4j usando la función híbrida junto con los keywords generados.
3. **Memoria ([SemanticMemory](file:///home/tom/histo-test/ne4j-histo_bak.py#731-911)):** A veces a esto se suma una consulta adicional sobre Qdrant para extraer recuerdos conversacionales pasados.

**¿Qué modelo se usa y dónde se ejecuta?**
Durante nuestra inicialización en [core/agent.py](file:///home/tom/histo-test/core/agent.py) definimos esto:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': self.device}
)
```
- **Modelo:** `all-MiniLM-L6-v2`. Es un modelo veloz y asombrosamente rápido con dimensionalidad 384 (`DIM_TEXTO`).
- **Conectividad:** Se ejecuta **LOCALMENTE**. No se envían las preguntas a un servidor de texto para vectorizar a través de API. El modelo se descarga usando HuggingFace Transformers la primera vez y corre apoyado de PyTorch en CPU o GPU.

### Situación de las imágenes:
Cuando subes una imagen en el [modo_interactivo](file:///home/tom/histo-test/ne4j-histo_bak.py#2207-2303), esta se procesa a la vez con:
- [UniWrapper](file:///home/tom/histo-test/ne4j-histo_bak.py#268-306) (MahmoodLab/UNI) -> Corre en **local** (produce 1024 floats).
- [PlipWrapper](file:///home/tom/histo-test/ne4j-histo_bak.py#238-267) (vinid/plip) -> Corre en **local** (produce 512 floats).

*Toda la vectorización es local. Lo único remotizado es la generación de texto racional que va hacia los servidores de la API Groq (Llama-4).*

---

## 3. Estado de [embed_query_con_reintento](file:///home/tom/histo-test/ne4j-histo_bak.py#169-185)

**¿Son código legado vestigial?**
**Sí, absolutamente.** En algún momento previo del código, parece que la generación de embeddings se mandaba a procesar a través de alguna API externa (por ejemplo OpenAI o Google). 

Las funciones [embed_query_con_reintento](file:///home/tom/histo-test/ne4j-histo_bak.py#169-185) y [embed_documents_con_reintento](file:///home/tom/histo-test/utils/resilience.py#77-96) en el fichero [utils/resilience.py](file:///home/tom/histo-test/utils/resilience.py) hacen un bloque `try/except` buscando los identificadores "429" (Límite de API / Rate limit) o "RESOURCE_EXHAUSTED". Como ahora hemos migrado toda la vectorización a HuggingFaceEmbeddings (`all-MiniLM-L6-v2`), este corre siempre en la máquina local sin límites de rate-limit. Por lo tanto, el control de errores `429/503` nunca se activará porque es código ejecutado on-premise (local).

---

## 4. Análisis de Entorno Colab: Clonación y Unificación

**¿Es factible clonar el repo e instalar dependencias OS como `tesseract`/`poppler` mediante `uv`?**
¡Es **vital** e **imprescindible**! Google Colab proporciona instancias efímeras (máquinas Ubuntu limpias en cada sesión nueva). Para que tu código modular funcione allí y no estalle en pedazos:
1. Necesitamos descargar el código de GitHub a la máquina de Colab (Clonar el Repo).
2. Necesitamos pre-instalar dependencias binarias a nivel de Sistema Operativo (`poppler-utils` y `tesseract-ocr`) usando `apt-get` para que las envolturas Pip `pdf2image` y `pytesseract` de [extractors/pdf.py](file:///home/tom/histo-test/extractors/pdf.py) puedan funcionar correctamente.
3. Usar el empaquetador `uv` es un enfoque moderno excelente para saltarse la lenta resolución de dependencias típica de Colab.

**¿Es buena idea tener la Ingestión (PipelineIngestion) y la Inferencia (modo_interactivo) en el mismo Notebook?**
Sí, es una arquitectura de **"Notebook Unificado"** fantástica.
Si estuvieran en notebooks separados, tendrías que duplicar el tedioso bloque inicial donde clonas el repositorio Github y donde importas las variables de los "Secretos de Colab". 

La mejor estructura presenta un solo cuaderno ([.ipynb](file:///home/tom/histo-test/GraphRAG_Colab_runner.ipynb)) con una organización celular clara:
*   **Celda 1 (Bootstrapping y Config):** Instala los paquetes Linux `apt-get`, ejecuta `uv`, clona el repositorio desde GitHub, e inyecta los `userdata.get()` estáticamente en `os.environ`.
*   **Celda 2 (ETL - Opcional):** Bloque donde se llama a `await PipelineIngestion().ejecutar()`. El usuario ejecuta esta celda *únicamente* cuando necesita precalentar y rellenar los índices Neo4j con PDFs nuevos.
*   **Celda 3 (Inferencia RAG - Iterativa):** Bloque donde se llama a `await modo_interactivo()`. El usuario la ejecuta frecuentemente para mantener el bucle de chat.

Ésta es, sin duda, la estructura que provee mejor UX (Developer/User Experience).
