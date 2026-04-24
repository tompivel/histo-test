# Implementation Plan: Conversational Chat

## Overview

Transformar el chat RAG de histología de un sistema pregunta-respuesta aislado a una experiencia conversacional natural. Se implementa en orden incremental: primero la infraestructura de memoria y detección, luego los cambios en el motor RAG, después el backend/API, y finalmente el frontend.

## Tasks

- [x] 1. Agregar `get_history_for_prompt()` a SemanticMemory y detección de anáforas
  - [x] 1.1 Implementar método `get_history_for_prompt(max_turns=5)` en la clase `SemanticMemory` de `ne4j-histo.py`
    - Leer `self.conversations[-max_turns:]` y formatear como `"Usuario: {query}\nAsistente: {response_truncada}"` por cada entrada
    - Truncar cada respuesta a 200 caracteres para no saturar el prompt
    - Retornar string vacío si no hay conversaciones
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [ ]* 1.2 Write property test: History windowing returns at most N entries
    - **Property 1: History windowing returns at most N most recent entries**
    - Usar Hypothesis para generar historiales de 0-50 entradas con max_turns variable
    - Verificar que retorna ≤N entradas y son las más recientes en orden cronológico
    - **Validates: Requirements 1.1, 1.4**

  - [ ]* 1.3 Write property test: History formatting preserves structure
    - **Property 2: History formatting preserves structure**
    - Verificar que cada entrada tiene exactamente un "Usuario:" y un "Asistente:" prefix
    - Verificar orden cronológico de las entradas
    - **Validates: Requirements 1.2**

- [x] 2. Implementar detección de solicitud de imagen y reescritura anafórica
  - [x] 2.1 Implementar `_detectar_solicitud_imagen(consulta)` en `AsistenteHistologiaNeo4j` en `ne4j-histo.py`
    - Función que recibe un string y retorna `bool`
    - Keywords: "mostrame", "mostrá", "imagen de", "foto de", "quiero ver", "dejame ver", "enseñame", "enseñá", "muéstrame", "ver imagen", "ver foto"
    - Comparar contra `consulta.lower()`
    - _Requirements: 4.1_

  - [x] 2.2 Implementar `_reescribir_consulta_con_contexto(consulta, historial)` en `AsistenteHistologiaNeo4j` en `ne4j-histo.py`
    - Detectar anáforas: "eso", "esa", "esto", "lo mismo", "sobre eso", "al respecto", "más sobre", "qué más", "y qué", "también", "además"
    - Si no hay anáforas, retornar consulta original sin cambios
    - Si hay anáforas, invocar LLM con prompt corto para reescribir la consulta como autocontenida
    - Manejar error del LLM retornando la consulta original (fallback seguro)
    - _Requirements: 1.3_

  - [ ]* 2.3 Write property test: Anaphora detection identifies referential queries
    - **Property 3: Anaphora detection identifies referential queries**
    - Generar strings con/sin keywords anafóricos usando Hypothesis
    - Verificar que la detección retorna True/False correctamente
    - **Validates: Requirements 1.3**

  - [ ]* 2.4 Write property test: Image request detection is keyword-driven
    - **Property 4: Image request detection is keyword-driven**
    - Generar strings con/sin keywords de imagen usando Hypothesis
    - Verificar que `_detectar_solicitud_imagen` retorna el booleano correcto
    - **Validates: Requirements 3.7, 4.1**

- [x] 3. Agregar campo `mostrar_imagenes` a AgentState y modificar `_nodo_inicializar`
  - [x] 3.1 Agregar `mostrar_imagenes: bool` al `AgentState` TypedDict en `ne4j-histo.py`
    - _Requirements: 4.4_

  - [x] 3.2 Modificar `_nodo_inicializar` en `ne4j-histo.py` para integrar historial, reescritura anafórica y detección de imagen
    - Obtener historial con `self.memoria.get_history_for_prompt(5)`
    - Llamar `_reescribir_consulta_con_contexto(consulta, historial)` y actualizar `state["consulta_texto"]` si se reescribió
    - Llamar `_detectar_solicitud_imagen(consulta)` y guardar resultado en `state["mostrar_imagenes"]`
    - Guardar el historial formateado en el state para uso posterior en el system prompt
    - _Requirements: 1.1, 1.3, 4.1, 4.2_

  - [x] 3.3 Inicializar `mostrar_imagenes=False` en el `initial_state` del método `consultar()`
    - _Requirements: 4.4_

- [x] 4. Checkpoint - Verificar lógica backend
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Modificar system prompts para prosa natural e inyectar historial conversacional
  - [x] 5.1 Modificar el system prompt de modo texto en `_nodo_generar_respuesta` de `ne4j-histo.py`
    - Reemplazar instrucciones de formato estructurado por instrucción de responder en prosa como un profesor explicando
    - Incluir instrucción explícita: "Respondé en prosa, como un profesor explicando en texto corrido. Evitá listas con bullets, numeración y formato estructurado rígido."
    - Inyectar bloque de historial conversacional antes del contexto RAG en el `content_parts`
    - Si hay historial, agregar instrucción de adaptar tono como continuación natural de la conversación
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [x] 5.2 Modificar el system prompt de modo multimodal en `_nodo_generar_respuesta` de `ne4j-histo.py`
    - Mismas instrucciones de prosa natural que el modo texto
    - Inyectar bloque de historial conversacional
    - Instruir integración narrativa de descripción visual con contenido del manual
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

- [x] 6. Propagar `mostrar_imagenes` desde RAG hasta la API y montar ruta estática
  - [x] 6.1 Agregar campo `mostrar_imagenes: bool = False` al modelo Pydantic `ChatResponse` en `server.py`
    - _Requirements: 4.4_

  - [x] 6.2 Modificar el endpoint `post_chat` en `server.py` para extraer `mostrar_imagenes` del resultado del grafo y pasarlo a `ChatResponse`
    - Leer `mostrar_imagenes` desde `trayectoria_neo4j.json` o desde el state final del grafo
    - Default a `False` si no está disponible
    - _Requirements: 4.4_

  - [x] 6.3 Montar ruta estática `/imagenes_extraidas/` en `server.py` usando `StaticFiles`
    - `app.mount("/imagenes_extraidas", StaticFiles(directory=str(IMAGENES_DIR)), name="imagenes_extraidas")`
    - Definir `IMAGENES_DIR = Path(__file__).parent / "imagenes_extraidas"`
    - _Requirements: 3.4_

  - [x] 6.4 Modificar `_nodo_finalizar` en `ne4j-histo.py` para incluir `mostrar_imagenes` en el JSON de trayectoria guardado en `trayectoria_neo4j.json`
    - _Requirements: 4.4_

- [x] 7. Checkpoint - Verificar backend completo
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implementar renderizado condicional de imágenes en frontend
  - [x] 8.1 Modificar la función `addMessage()` en `client/app.js` para renderizar imágenes cuando `mostrar_imagenes=true`
    - Cuando `role === 'assistant'` y `metadata.mostrar_imagenes === true` y `metadata.imagenes_recuperadas.length > 0`:
      - Crear contenedor `div.retrieved-images`
      - Por cada filename en `imagenes_recuperadas`, crear `<figure>` con `<img src="/imagenes_extraidas/${filename}">` y `<figcaption>` con el nombre del archivo
      - Agregar `onclick` para abrir imagen en nueva pestaña
      - Agregar handler `onerror` para ocultar imágenes que no cargan
    - Cuando `mostrar_imagenes` es `false` o ausente, no renderizar imágenes aunque `imagenes_recuperadas` tenga datos
    - _Requirements: 3.3, 3.5, 3.7, 4.3_

  - [x] 8.2 Agregar estilos CSS para imágenes recuperadas en `client/style.css`
    - Estilos para `.retrieved-images`, `.retrieved-image-figure`, `.retrieved-image`, `figcaption`
    - Grid o flex layout para múltiples imágenes
    - Bordes, border-radius y hover effects consistentes con el tema dark glassmorphism existente
    - Responsive para pantallas pequeñas
    - _Requirements: 3.3, 3.5_

- [x] 9. Final checkpoint - Verificar integración completa
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis (Python)
- El lenguaje de implementación es Python (backend) y JavaScript (frontend), según el código existente del proyecto
- Los system prompts se modifican in-place en `_nodo_generar_respuesta`, no se crean archivos nuevos
