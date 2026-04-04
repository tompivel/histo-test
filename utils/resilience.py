import asyncio
import time

async def invoke_with_retry(llm, messages, max_retries=5):
    """
    Wraps asynchronous LangChain LLM requests handling exponential backoff 
    when rate limits or server exhausted resources occur.
    """
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ API Quota Limit/Server Busy (429/503) - retrying in {espera}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(espera)
                else:
                    raise e
            else:
                raise e

def invoke_with_retry_sync(llm, messages, max_retries=5):
    """
    Identical behavior to `invoke_with_retry` but paralyzing the thread synchronously.
    Ideal for CLI scripts and background fetchers.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ API Quota Limit/Server Busy (429/503) - retrying in {espera}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e

def embed_query_with_retry(embeddings, texto: str, max_retries=5):
    """
    Executes transaction backoff over local or API individual text inference. (Synchronous).
    """
    for attempt in range(max_retries):
        try:
            return embeddings.embed_query(texto)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ API Quota Limit in Embeddings (429/503) - retrying in {espera}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e

def embed_documents_with_retry(embeddings, textos: list, max_retries=5):
    """
    Wraps batch functions for large collections providing resilience against HF/Vertex API pauses.
    """
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(textos)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    espera = 15 * (attempt + 1)
                    print(f"   ⚠️ API Quota Limit in Embeddings (429/503) - retrying in {espera}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(espera)
                else:
                    raise e
            else:
                raise e
