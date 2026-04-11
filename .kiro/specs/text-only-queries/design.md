# Design Document: Text-Only Queries

## Overview

Esta funcionalidad extiende el sistema RAG multimodal de histología (v4.2) para optimizar el procesamiento de consultas que no incluyen imágenes. El sistema actual ya implementa un flujo bifurcado en LangGraph que detecta la presencia de imágenes, pero requiere refinamiento y completitud según los requisitos formalizados.

### Objetivos del Diseño

1. **Flujo Optimizado**: Implementar un router condicional robusto que detecte el modo de consulta (texto puro vs multimodal) y dirija el flujo apropiadamente, omitiendo nodos innecesarios.

2. **Umbrales Diferenciados**: Aplicar umbrales de similitud más permisivos (0.45) para consultas de texto puro, reconociendo que sin contexto visual la búsqueda textual requiere mayor flexibilidad.

3. **Prompts Especializados**: Utilizar system prompts diferenciados que instruyan al LLM a generar respuestas enciclopédicas basadas exclusivamente en el manual, sin referencias a imágenes del usuario.

4. **Manejo Amable de Contexto Insuficiente**: Cuando no se encuentra información relevante en modo texto, proporcionar mensajes útiles con sugerencias y muestra del temario disponible.

5. **Integración con Memoria**: Mantener coherencia en la memoria semántica independientemente del modo, permitiendo transiciones fluidas entre consultas con y sin imagen.

### Alcance

**Incluido:**
- Modificaciones al grafo LangGraph para routing condicional
- Ajustes en nodos de procesamiento para omitir pasos relacionados con imágenes
- Implementación de umbrales diferenciados según modo
- System prompts específicos para modo texto
- Mensajes amables para contexto insuficiente
- Registro completo de trayectoria para debugging

**Excluido:**
- Cambios en los modelos de embeddings (HuggingFace, UNI, PLIP)
- Modificaciones a la estructura de Neo4j
- Cambios en la extracción de imágenes de PDFs
- Alteraciones al clasificador semántico (ya funciona para ambos modos)

## Architecture

### Flujo de Procesamiento

El sistema implementa un grafo de estado en LangGraph con routing condicional basado en la presencia de imágenes:

```
START
  ↓
Inicializar
  ↓
[Router: _route_por_modo]
  ├─→ con_imagen → ProcesarImagen → Clasificar
  └─→ solo_texto → Clasificar
       ↓
[Router: _route_por_temario]
  ├─→ en_temario → GenerarConsulta
  └─→ fuera_temario → FueraTemario → END
       ↓
BuscarNeo4j
  ↓
FiltrarContexto
  ↓
[Router: _route_analisis_comparativo]
  ├─→ con_imagen → AnalisisComparativo → GenerarRespuesta
  └─→ sin_imagen → GenerarRespuesta
       ↓
Finalizar
  ↓
END
```

### Decisiones de Routing

**Router 1: _route_por_modo**
- **Entrada**: `state["imagen_path"]`, `memoria.tiene_imagen_previa()`
- **Salida**: `"con_imagen"` | `"solo_texto"`
- **Lógica**: 
  - Si `imagen_path` existe Y es válida → `"con_imagen"`
  - Si memoria tiene imagen activa → `"con_imagen"`
  - Caso contrario → `"solo_texto"`

**Router 2: _route_por_temario**
- **Entrada**: `state["tema_valido"]`
- **Salida**: `"en_temario"` | `"fuera_temario"`
- **Lógica**: Basado en clasificador semántico

**Router 3: _route_analisis_comparativo**
- **Entrada**: `state["tiene_imagen"]`, `state["imagen_path"]`
- **Salida**: `"con_imagen"` | `"sin_imagen"`
- **Lógica**: Solo ejecuta análisis comparativo si hay imagen disponible

### Componentes Modificados

1. **AsistenteHistologiaNeo4j._route_por_modo**: Router condicional principal
2. **AsistenteHistologiaNeo4j._nodo_filtrar_contexto**: Aplica umbrales diferenciados
3. **AsistenteHistologiaNeo4j._nodo_generar_respuesta**: System prompts diferenciados
4. **AsistenteHistologiaNeo4j._nodo_clasificar**: Omite análisis visual en modo texto
5. **AsistenteHistologiaNeo4j._nodo_generar_consulta**: Omite consulta visual en modo texto
6. **SemanticMemory**: Maneja imagen_activa_path=None para modo texto

## Components and Interfaces

### 1. Router Condicional

**Interfaz:**
```python
def _route_por_modo(self, state: AgentState) -> str:
    """
    Determina el modo de procesamiento basado en disponibilidad de imagen.
    
    Args:
        state: Estado del grafo con imagen_path y contexto de memoria
        
    Returns:
        "con_imagen" si hay imagen nueva o activa en memoria
        "solo_texto" si no hay imagen disponible
    """
```

**Implementación:**
```python
def _route_por_modo(self, state: AgentState) -> str:
    imagen_path = state.get("imagen_path")
    tiene_imagen_nueva = imagen_path and os.path.exists(imagen_path)
    tiene_imagen_memoria = self.memoria and self.memoria.tiene_imagen_previa()
    
    if tiene_imagen_nueva or tiene_imagen_memoria:
        return "con_imagen"
    
    print("📝 Modo solo texto — saltando procesamiento de imagen")
    return "solo_texto"
```

### 2. Filtrado con Umbrales Diferenciados

**Interfaz:**
```python
async def _nodo_filtrar_contexto(self, state: AgentState) -> AgentState:
    """
    Filtra resultados de búsqueda aplicando umbrales según modo.
    
    Umbrales:
        - Modo texto: umbral_texto=0.45 (más permisivo)
        - Modo multimodal: umbral_texto=0.6, umbral_imagen=0.45
        
    Returns:
        state actualizado con resultados_validos y contexto_suficiente
    """
```

**Lógica de Umbrales:**
```python
es_solo_texto = not state.get("tiene_imagen", False)
umbral_texto = 0.45 if es_solo_texto else 0.6
umbral_imagen = 0.45  # Constante para imágenes

for resultado in state["resultados_busqueda"]:
    if resultado["tipo"] == "texto":
        if resultado["similitud"] >= umbral_texto:
            validos.append(resultado)
    elif resultado["tipo"] == "imagen":
        if resultado["similitud"] >= umbral_imagen:
            if os.path.exists(resultado["imagen_path"]):
                validos.append(resultado)
```

### 3. System Prompts Diferenciados

**Prompt para Modo Texto:**
```python
system_prompt_texto = """
Eres un asistente experto de histología. Respondés consultas de texto 
basándote EXCLUSIVAMENTE en el contenido del manual/base de datos.

REGLAS FUNDAMENTALES:
1. FUENTE DE VERDAD: Usá SOLO la información de las SECCIONES DEL MANUAL proporcionadas.
2. CITAS: Citá las fuentes con [Manual: archivo].
3. NO inventes información que no esté en las secciones proporcionadas.
4. Si la información es parcial, indicá qué partes provienen del manual y cuáles no están disponibles.
5. No diagnósticos clínicos salvo que estén explícitos en el manual.

ESTRUCTURA DE RESPUESTA:
1. Respuesta directa a la consulta basada en el manual
2. Características histológicas relevantes según la base de datos
3. Fuentes y referencias del manual
4. Conclusión
"""
```

**Prompt para Modo Multimodal:**
```python
system_prompt_multimodal = """
Eres un asistente de histología. Responde SOLO con el contenido del manual 
o la imagen visible en el chat.

REGLAS FUNDAMENTALES:
1. PRIORIDAD ABSOLUTA: La DESCRIPCIÓN TEXTUAL DEL MANUAL es la fuente de verdad.
2. Cita: [Manual: archivo] | [Imagen: archivo]
3. Para cada 'IMAGEN DE REFERENCIA', indica el nombre y la descripción textual del manual.
4. NO hagas diagnósticos propios basados en tu interpretación visual.
5. No diagnósticos clínicos salvo que estén explícitos.

ESTRUCTURA:
1. Análisis de la consulta basado en la imagen del usuario (si la hay)
2. VALIDACIÓN: Revisa el 'ANÁLISIS COMPARATIVO'...
3. Características histológicas según la base de datos
4. Conclusión y confianza
"""
```

### 4. Manejo de Contexto Insuficiente

**Interfaz:**
```python
def _generar_mensaje_sin_contexto_texto(self, temario: List[str]) -> str:
    """
    Genera mensaje amable cuando no hay contexto suficiente en modo texto.
    
    Args:
        temario: Lista de temas disponibles en el manual
        
    Returns:
        Mensaje formateado con sugerencias y muestra del temario
    """
```

**Implementación:**
```python
muestra = "\n".join(f"  • {t}" for t in temario[:15])
if len(temario) > 15:
    muestra += f"\n  ... y {len(temario)-15} más"

mensaje = f"""
⚠️ **No encontré información específica sobre eso en el manual**

La consulta es válida pero no encontré contenido suficiente en la 
base de datos del manual para responderla con precisión.

**Temas disponibles en el manual (muestra):**
{muestra}

Podés intentar:
- Reformular la pregunta con términos más específicos
- Preguntar sobre alguno de los temas listados arriba
- Subir una imagen histológica para análisis visual
"""
```

### 5. Integración con Memoria Semántica

**Modificaciones a SemanticMemory:**

```python
class SemanticMemory:
    def add_interaction(self, query: str, response: str):
        """
        Guarda interacción independientemente del modo.
        Para modo texto: imagen=None
        """
        self.turno_actual += 1
        self.conversations.append({
            "query": query,
            "response": response,
            "turno": self.turno_actual,
            "imagen": self.imagen_activa_path  # None en modo texto
        })
        
        # Guardar en Qdrant cada 5 turnos
        if self.turno_actual % 5 == 0:
            self._guardar_memoria_qdrant()
    
    def _guardar_memoria_qdrant(self):
        """
        Guarda resumen con embeddings.
        Para modo texto: embedding_uni y embedding_plip son vectores cero.
        """
        emb_texto = embed_query_con_reintento(self.embeddings, resumen)
        
        # Embeddings visuales (cero si no hay imagen)
        emb_uni = [0.0] * DIM_IMG_UNI
        emb_plip = [0.0] * DIM_IMG_PLIP
        
        if self.imagen_activa_path and os.path.exists(self.imagen_activa_path):
            if self.uni:
                emb_uni = self.uni.embed_image(self.imagen_activa_path).tolist()
            if self.plip:
                emb_plip = self.plip.embed_image(self.imagen_activa_path).tolist()
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "texto": emb_texto,
                "uni": emb_uni,
                "plip": emb_plip
            },
            payload={
                "resumen": resumen,
                "turno_fin": self.turno_actual,
                "tiene_imagen": self.imagen_activa_path is not None,
                "imagen_path": self.imagen_activa_path
            }
        )
        self.qdrant.upsert(collection_name=self.collection_name, points=[point])
```

## Data Models

### AgentState (TypedDict)

Estado del grafo LangGraph con campos relevantes para modo texto:

```python
class AgentState(TypedDict):
    # Campos existentes
    consulta_texto: str
    imagen_path: Optional[str]
    tiene_imagen: bool
    imagen_es_nueva: bool
    
    # Embeddings
    texto_embedding: Optional[List[float]]  # Siempre generado
    imagen_embedding_uni: Optional[List[float]]  # None en modo texto
    imagen_embedding_plip: Optional[List[float]]  # None en modo texto
    
    # Análisis
    analisis_visual: Optional[str]  # None en modo texto
    terminos_busqueda: str
    entidades_consulta: Dict[str, List[str]]
    
    # Búsqueda
    consulta_busqueda_texto: str
    consulta_busqueda_visual: str  # Vacío en modo texto
    resultados_busqueda: List[Dict[str, Any]]
    resultados_validos: List[Dict[str, Any]]
    
    # Contexto
    contexto_suficiente: bool
    contexto_documentos: str
    
    # Respuesta
    respuesta_final: str
    analisis_comparativo: Optional[str]  # None en modo texto
    estructura_identificada: Optional[str]  # None en modo texto
    
    # Metadata
    trayectoria: List[Dict[str, Any]]
    temario: List[str]
    tema_valido: bool
    tema_encontrado: Optional[str]
```

### Trayectoria JSON

Estructura del archivo `trayectoria_neo4j.json`:

```json
{
  "trayectoria": [
    {
      "nodo": "Inicializar",
      "tiempo": 0.05
    },
    {
      "nodo": "Clasificar",
      "tema_valido": true,
      "tema_encontrado": "tejido epitelial",
      "entidades": {
        "tejidos": ["epitelio"],
        "estructuras": ["célula"],
        "tinciones": ["h&e"]
      },
      "similitud_dominio": 0.78,
      "metodo_clasificacion": "semantico_imagebind",
      "tiempo": 0.32
    },
    {
      "nodo": "GenerarConsulta",
      "query": "tejido epitelial características",
      "tiempo": 0.18
    },
    {
      "nodo": "BuscarNeo4j",
      "hits": 12,
      "tiempo": 0.45
    },
    {
      "nodo": "FiltrarContexto",
      "hits_validos": 8,
      "imgs": 0,
      "modo": "solo_texto",
      "tiempo": 0.08
    },
    {
      "nodo": "GenerarRespuesta",
      "contexto_suficiente": true,
      "modo": "solo_texto",
      "tiempo": 2.15
    }
  ],
  "estructura_identificada": null,
  "imagenes_recuperadas": []
}
```

### Configuración de Umbrales

```python
# Constantes globales
SIMILARITY_THRESHOLD = 0.45  # Base para imágenes

# Umbrales dinámicos en _nodo_filtrar_contexto
def get_umbral_texto(es_solo_texto: bool) -> float:
    return 0.45 if es_solo_texto else 0.6

def get_umbral_imagen() -> float:
    return 0.45  # Constante
```

### Pesos de Búsqueda Híbrida

```python
# En Neo4jClient.busqueda_hibrida
tiene_imagen = imagen_embedding_uni is not None or imagen_embedding_plip is not None

if tiene_imagen:
    # Modo multimodal
    agregar(res_texto, 0.40)
    agregar(res_uni, 0.70)
    agregar(res_plip, 0.70)
    agregar(res_ent, 0.60)
else:
    # Modo texto puro
    agregar(res_texto, 0.80)  # Texto domina
    agregar(res_uni, 0.20)    # Poco relevante
    agregar(res_plip, 0.20)   # Poco relevante
    agregar(res_ent, 0.60)    # Entidades importantes

agregar(res_vec, 0.20)  # Vecindad siempre igual
```

