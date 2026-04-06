# Port 3 Features + Retrieval Bugfix

## Contexto

Porting features from `~/Escritorio/CetecFiuba/Grupo1.py` (v4.1 con ImageBind) al nuevo repositorio `histo-test/ne4j-histo.py` (v4.2 con UNI+PLIP+Groq).

---

## Cambios Propuestos

### Feature 1 — RecursiveCharacterTextSplitter (800 chars, 150 overlap)

#### [MODIFY] [ne4j-histo.py](file:///home/francisco/Escritorio/IA/histo-test/ne4j-histo.py)

**Problema actual:** El chunking en la nueva arquitectura (línea 2033-2034) es un split ingenuo de 500 chars sin overlap:
```python
def _chunks(self, texto: str, size: int = 500) -> List[str]:
    return [texto[i:i+size] for i in range(0, len(texto), size)]
```

**Solución:** Reemplazar con la lógica recursiva de `Grupo1.py` (líneas 1696-1744), que usa separadores jerárquicos `["\n\n", "\n", ". ", " ", ""]` con `chunk_size=800` y `chunk_overlap=150`.

> [!IMPORTANT]
> Esto **cambiará los chunk IDs** (`chunk_{fuente}_{i}`) y la cantidad de chunks por PDF. Si ya hay datos indexados en Neo4j, será necesario re-indexar (`--reindex --force`).

---

### Feature 2 — Extracción de Tablas Multimodales con LLM

#### [MODIFY] [ne4j-histo.py](file:///home/francisco/Escritorio/IA/histo-test/ne4j-histo.py)

**Problema actual:** El `ExtractorImagenesPDF` actual (línea 1040) **no extrae tablas**. El viejo `Grupo1.py` usaba un LLM multimodal (Gemini) para detectar tablas en cada página y guardarlas como nodos `:Tabla` en Neo4j con su contenido en Markdown.

**Solución:**
1. Agregar el método `_detectar_y_extraer_tabla()` al `ExtractorImagenesPDF` (adaptado para Groq/Llama que **no acepta imágenes** — ver pregunta abierta).
2. Agregar constraint `Tabla` al schema de Neo4j.
3. Agregar `upsert_tabla()` al `Neo4jClient`.
4. En `indexar_en_neo4j()`, al procesar cada imagen de página, si hay tabla detectada, crear el nodo `:Tabla`.

> [!WARNING]
> **Groq con Llama 4 Scout soporta visión multimodal**, por lo que podemos enviar la imagen de la página al LLM para detectar tablas. Sin embargo, hay que confirmar que el modelo `meta-llama/llama-4-scout-17b-16e-instruct` acepte `image_url` content parts vía Groq.

---

### Feature 3 — Normalización SNOMED/FMA en Nodos Tejido/Estructura

#### [MODIFY] [ne4j-histo.py](file:///home/francisco/Escritorio/IA/histo-test/ne4j-histo.py)

**Problema actual:** El `upsert_chunk` de la nueva arquitectura (líneas 434-445) trata tejidos y estructuras como strings planos:
```python
for tejido in entidades.get("tejidos", []):
    await self.run("""
        MERGE (t:Tejido {nombre: $nombre})
        ...
    """, {"nombre": tejido, "chunk_id": chunk_id})
```

El viejo `Grupo1.py` (líneas 242-263) extraía `snomed_id` y `fma_id` como diccionarios y los guardaba como propiedades.

**Solución:**
1. Modificar `ExtractorEntidades.extraer_de_texto()` para que retorne diccionarios con `nombre`, `snomed_id`, `fma_id` (como en Grupo1.py líneas 888-909).
2. Modificar `Neo4jClient.upsert_chunk()` para que acepte diccionarios y haga `SET t.snomed_id = $snomed_id, t.fma_id = $fma_id`.
3. Adaptar el `extraer_de_texto_sync()` fallback para que siga funcionando (genera strings, no dicts).

---

### Feature 4 — BUGFIX: Re-ranking Híbrido para Exact Image Match

#### [MODIFY] [ne4j-histo.py](file:///home/francisco/Escritorio/IA/histo-test/ne4j-histo.py)

**Problema:** Cuando el usuario sube una captura que es **literalmente una imagen extraída del manual** (o un crop/screenshot de la misma), esa imagen debería rankear en Top-1. Actualmente queda en posición ~4 porque:
1. La búsqueda vectorial UNI/PLIP devuelve scores relativamente uniformes entre imágenes similares del mismo tejido.
2. Los pesos de fusión (`0.80 texto`, `0.50 UNI`, `0.50 PLIP`) diluyen una coincidencia visual exacta con mucho ruido textual.

**Solución — Re-ranking en 3 pasos:**

1. **Detección de Near-Duplicate:** Después de obtener resultados de `busqueda_vectorial` para UNI y PLIP, detectar si algún resultado tiene score ≥ 0.95 en **ambos** modelos (UNI y PLIP). Eso indica near-duplicate (misma imagen o crop).

2. **Boost de Near-Duplicates:** En `busqueda_hibrida()`, si un resultado es near-duplicate, aplicar un boost multiplicativo (ej: `similitud *= 2.0`) para que suba al Top-1 tras la fusión.

3. **Validación Semántica por Grafo:** Para los top-4 resultados de imagen, usar las relaciones del grafo para verificar coherencia: si la imagen está vinculada al mismo Tejido/Estructura que menciona la consulta, dar bonus adicional.

**Implementación concreta en `busqueda_hibrida()`:**

```python
# Paso extra: identificar near-duplicates (UNI ∩ PLIP con score alto)
near_dup_ids = set()
if res_uni and res_plip:
    uni_scores = {r["id"]: r["similitud"] for r in res_uni}
    plip_scores = {r["id"]: r["similitud"] for r in res_plip}
    for img_id in uni_scores:
        if img_id in plip_scores:
            if uni_scores[img_id] >= 0.95 and plip_scores[img_id] >= 0.95:
                near_dup_ids.add(img_id)

# En agregar(), aplicar boost:
def agregar(resultados, peso):
    for r in resultados:
        ...
        if r.get("id") in near_dup_ids:
            sim_ponderada *= 2.0  # Near-duplicate boost
        ...
```

---

## Resumen de Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `ne4j-histo.py` | Chunking recursivo, tabla extraction, SNOMED IDs, re-ranking |
| `server.py` | Sin cambios (los cambios son transparentes) |

## Open Questions

> [!IMPORTANT]
> 1. **¿Tu modelo Groq (Llama 4 Scout) acepta imágenes vía `image_url`?** Si no, la extracción de tablas podría hacerse con OCR + heurística en lugar de LLM multimodal. ¿Preferís que use otro enfoque?
> 2. **El chunking con overlap de 150 chars puede generar ~60% más chunks** que el actual de 500 sin overlap. Esto impacta el tiempo de indexación y las API calls de embeddings. ¿Está bien?
> 3. **Para el re-ranking, ¿el umbral de 0.95 para near-duplicate te parece correcto?** Puede ajustarse si tus imágenes de prueba son crops con bordes diferentes.

## Verification Plan

### Automated Tests
- Re-indexar con `--reindex --force` después de los cambios.
- Verificar en la consola que los chunks ahora son ~800 chars.
- Verificar que nodos `:Tabla` aparezcan en Neo4j.
- Verificar que nodos `:Tejido` tengan propiedad `snomed_id`.

### Manual Verification
- Subir una captura del manual como imagen de consulta y verificar que rankee Top-1.
- Comparar tiempos de respuesta antes vs después.
