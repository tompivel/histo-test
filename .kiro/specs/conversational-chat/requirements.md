# Requirements Document

## Introduction

Mejora del chat del sistema RAG de histología para lograr una experiencia conversacional natural. Actualmente el sistema tiene SemanticMemory con Qdrant pero las respuestas no fluyen como una conversación real: no consideran el contexto previo de forma coherente, se formatean con listas y bullets rígidos, y no devuelven imágenes del contenido indexado cuando el usuario las solicita. Esta feature aborda tres ejes: memoria conversacional real, respuestas en prosa natural, y devolución de imágenes de la base de datos.

## Glossary

- **Chat_Backend**: El endpoint `/api/chat` de FastAPI en `server.py` que recibe consultas y retorna `ChatResponse`.
- **RAG_Engine**: El motor de consulta implementado en `ne4j-histo.py` con LangGraph, Neo4j y embeddings UNI/PLIP.
- **SemanticMemory**: La clase `SemanticMemory` en `ne4j-histo.py` que mantiene historial de conversación y persistencia en Qdrant.
- **LLM**: El modelo de lenguaje (Groq/Llama) usado para generar respuestas en el nodo `generar_respuesta` del grafo LangGraph.
- **Chat_Frontend**: La interfaz de usuario en `client/app.js` e `client/index.html` que renderiza mensajes y maneja interacción.
- **Imagenes_Indexadas**: Las imágenes en `imagenes_extraidas/` indexadas como nodos `:Imagen` en Neo4j con paths, captions y embeddings.
- **Conversation_History**: El historial de intercambios previos entre usuario y asistente dentro de una sesión de chat.
- **Prose_Response**: Una respuesta en texto corrido, como la explicación natural de un profesor, sin listas con bullets ni formato estructurado rígido.
- **Image_Request**: Una consulta del usuario que solicita explícitamente ver una imagen del contenido indexado.

## Requirements

### Requirement 1: Memoria Conversacional en el System Prompt

**User Story:** Como estudiante de histología, quiero que el asistente recuerde lo que hablamos antes y responda teniendo en cuenta el contexto de la conversación, para que la interacción fluya como un diálogo natural con un profesor.

#### Acceptance Criteria

1. WHEN el usuario envía una consulta, THE RAG_Engine SHALL inyectar el Conversation_History de los últimos 5 intercambios en el system prompt del LLM como contexto conversacional.
2. WHEN el Conversation_History contiene intercambios previos, THE LLM SHALL recibir un bloque de historial formateado con pares "Usuario: ... / Asistente: ..." antes del contexto RAG.
3. WHEN el usuario hace una pregunta de seguimiento que referencia temas previos (por ejemplo "¿y qué más me podés decir sobre eso?"), THE RAG_Engine SHALL resolver las referencias anafóricas usando el Conversation_History para generar la consulta de búsqueda.
4. WHEN el Conversation_History supera 5 intercambios, THE SemanticMemory SHALL mantener solo los 5 intercambios más recientes en el historial directo inyectado al prompt.
5. WHEN el usuario inicia una nueva sesión de chat, THE SemanticMemory SHALL comenzar con un Conversation_History vacío.

### Requirement 2: Respuestas en Prosa Natural

**User Story:** Como estudiante de histología, quiero que las respuestas del asistente sean explicaciones en texto corrido como las de un profesor, para que pueda leer y entender el contenido de forma fluida sin formato rígido.

#### Acceptance Criteria

1. THE LLM SHALL generar respuestas en formato de Prose_Response, usando párrafos de texto corrido en lugar de listas con bullets o numeración.
2. THE LLM SHALL recibir en el system prompt una instrucción explícita de responder como un profesor explicando en prosa, evitando listas, bullets y formato estructurado rígido.
3. WHEN el RAG_Engine genera una respuesta en modo solo texto, THE LLM SHALL producir una explicación narrativa que conecte los conceptos de forma fluida entre párrafos.
4. WHEN el RAG_Engine genera una respuesta en modo multimodal, THE LLM SHALL producir una explicación narrativa que integre la descripción visual con el contenido del manual en prosa continua.
5. WHEN el Conversation_History contiene intercambios previos, THE LLM SHALL adaptar el tono de la respuesta para que fluya como continuación natural de la conversación, evitando repetir información ya proporcionada.

### Requirement 3: Devolución de Imágenes del Contenido Indexado

**User Story:** Como estudiante de histología, quiero poder pedir imágenes del contenido (por ejemplo "mostrame una imagen de tejido epitelial") y que el sistema me las muestre en el chat, para complementar visualmente las explicaciones textuales.

#### Acceptance Criteria

1. WHEN el usuario envía una Image_Request explícita, THE RAG_Engine SHALL buscar imágenes relevantes en los nodos `:Imagen` de Neo4j usando búsqueda vectorial por texto.
2. WHEN el RAG_Engine encuentra Imagenes_Indexadas relevantes para una Image_Request, THE Chat_Backend SHALL incluir los paths de las imágenes en el campo `imagenes_recuperadas` de la respuesta.
3. WHEN la respuesta del Chat_Backend contiene `imagenes_recuperadas` con elementos, THE Chat_Frontend SHALL renderizar las imágenes dentro del mensaje del asistente como elementos visuales clicables.
4. THE Chat_Backend SHALL servir las Imagenes_Indexadas desde el directorio `imagenes_extraidas/` mediante una ruta estática accesible por el Chat_Frontend.
5. WHEN el Chat_Frontend renderiza una imagen recuperada, THE Chat_Frontend SHALL mostrar el nombre del archivo como caption debajo de la imagen.
6. IF el RAG_Engine no encuentra Imagenes_Indexadas relevantes para una Image_Request, THEN THE LLM SHALL indicar en la respuesta que no se encontraron imágenes disponibles para esa consulta.
7. WHEN el usuario no solicita imágenes explícitamente, THE RAG_Engine SHALL continuar retornando imágenes recuperadas en el campo `imagenes_recuperadas` sin que el Chat_Frontend las renderice de forma prominente, manteniendo el comportamiento actual de metadata.

### Requirement 4: Detección de Solicitud de Imagen

**User Story:** Como estudiante de histología, quiero que el sistema detecte cuándo estoy pidiendo una imagen del contenido, para que solo me muestre imágenes cuando realmente las necesito.

#### Acceptance Criteria

1. WHEN el usuario incluye en su consulta frases como "mostrame", "mostrá", "imagen de", "foto de", "quiero ver", "dejame ver" o "enseñame", THE RAG_Engine SHALL clasificar la consulta como Image_Request.
2. WHEN una consulta es clasificada como Image_Request, THE RAG_Engine SHALL priorizar la búsqueda de nodos `:Imagen` en Neo4j además de la búsqueda textual normal.
3. WHEN una consulta no es clasificada como Image_Request, THE Chat_Frontend SHALL ocultar las imágenes recuperadas del renderizado del mensaje, mostrando solo la respuesta textual.
4. THE RAG_Engine SHALL incluir un campo booleano `mostrar_imagenes` en la respuesta para indicar al Chat_Frontend si debe renderizar las imágenes recuperadas.
