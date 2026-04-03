import asyncio
import time

async def invoke_con_reintento(llm, messages, max_retries=5):
    """
    Envuelve las peticiones asíncronas a un LLM en LangChain controlando el backoff
    exponencial cuando ocurre límite de tasa o agotamiento de recursos del servidor.

    Args:
        llm: Instancia LangChain configurada para ainvoke.
        messages (list): Lista de diccionarios/objetos HumanMessage o SystemMessage.
        max_retries (int, opcional): Número máximo de intentos.

    Returns:
        BaseMessage: Resultado exitoso de la invocación LLM.

    Raises:
        Exception: Repropaga la falla original si excede los intentos.
    """
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
    """
    Idéntico comportamiento que `invoke_con_reintento` pero paralizando el hilo síncronamente.
    Ideal para scripts de línea de comando y background fetchers (ej. extracciones offline).
    """
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

# vestigial legacy code
def embed_query_con_reintento(embeddings, texto: str, max_retries=5):
    """
    Ejecuta backoff transaccional sobre la inferencia local o de API de texto individual 
    para modelos convertidores a vectores latentes. (Síncrono).
    """
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

# vestigial legacy code
def embed_documents_con_reintento(embeddings, textos: list, max_retries=5):
    """
    Envuelve funciones batch de incrustación de grandes colecciones de documentos
    proveyendo resistencia contra pausas del servidor API de HuggingFace/Vertex. (Síncrono).
    """
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
