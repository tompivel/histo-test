# Requirements Document

## Introduction

El sistema RAG multimodal de histología actualmente está diseñado principalmente para trabajar con consultas que incluyen imágenes histológicas. La funcionalidad de consultas de texto puro existe parcialmente en la versión 4.2 con un flujo bifurcado en LangGraph, pero requiere mejoras para garantizar que las consultas sin imágenes sean procesadas de manera óptima y proporcionen respuestas precisas basadas exclusivamente en el contenido textual del manual.

Esta funcionalidad permitirá a los usuarios realizar consultas teóricas sobre histología sin necesidad de proporcionar imágenes, aprovechando la base de conocimiento textual indexada en Neo4j y los embeddings semánticos de HuggingFace.

## Glossary

- **Sistema_RAG**: El sistema de Generación Aumentada por Recuperación (Retrieval-Augmented Generation) multimodal de histología
- **LangGraph**: Framework de orquestación de flujo de trabajo basado en grafos de estado
- **Neo4j**: Base de datos de grafos utilizada para almacenar y recuperar conocimiento histológico
- **Consulta_Texto_Puro**: Consulta del usuario que no incluye ninguna imagen, solo texto
- **Modo_Texto**: Modo de operación del sistema cuando procesa consultas sin imágenes
- **Modo_Multimodal**: Modo de operación del sistema cuando procesa consultas con imágenes
- **Umbral_Similitud**: Valor numérico que determina si un resultado de búsqueda es suficientemente relevante
- **Router**: Componente condicional en LangGraph que decide el camino del flujo según el tipo de consulta
- **Nodo_Procesamiento**: Paso individual en el grafo de LangGraph que ejecuta una operación específica
- **Embedding_Texto**: Representación vectorial del texto generada por HuggingFace (dimensión 384)
- **Contexto_RAG**: Información recuperada de la base de datos que se utiliza para generar la respuesta
- **Memoria_Semantica**: Componente que mantiene el historial de conversación y contexto entre turnos

## Requirements

### Requirement 1: Detección de Modo de Consulta

**User Story:** Como usuario del sistema, quiero que el sistema detecte automáticamente si mi consulta incluye o no una imagen, para que procese mi solicitud de manera apropiada.

#### Acceptance Criteria

1. WHEN una consulta no incluye imagen_path Y la Memoria_Semantica no tiene imagen activa, THEN THE Router SHALL retornar "solo_texto"
2. WHEN una consulta incluye imagen_path válida O la Memoria_Semantica tiene imagen activa, THEN THE Router SHALL retornar "con_imagen"
3. THE Sistema_RAG SHALL registrar en trayectoria el modo detectado con el valor "solo_texto" o "multimodal"
4. WHEN el modo es "solo_texto", THEN THE Sistema_RAG SHALL omitir el Nodo_Procesamiento "procesar_imagen"
5. WHEN el modo es "solo_texto", THEN THE Sistema_RAG SHALL omitir el Nodo_Procesamiento "analisis_comparativo"

### Requirement 2: Búsqueda Optimizada para Texto Puro

**User Story:** Como usuario realizando una consulta teórica, quiero que el sistema utilice umbrales de similitud apropiados para texto puro, para obtener resultados relevantes sin requerir una imagen.

#### Acceptance Criteria

1. WHEN el Modo_Texto está activo, THEN THE Sistema_RAG SHALL utilizar Umbral_Similitud de 0.45 para resultados de tipo texto
2. WHEN el Modo_Multimodal está activo, THEN THE Sistema_RAG SHALL utilizar Umbral_Similitud de 0.6 para resultados de tipo texto
3. THE Sistema_RAG SHALL generar Embedding_Texto de la consulta utilizando HuggingFace embeddings
4. THE Sistema_RAG SHALL ejecutar búsqueda vectorial en Neo4j usando el índice INDEX_TEXTO con el Embedding_Texto
5. THE Sistema_RAG SHALL ejecutar búsqueda por entidades en Neo4j usando tejidos, estructuras y tinciones extraídas
6. WHEN el Modo_Texto está activo, THEN THE Sistema_RAG SHALL asignar peso 0.80 a resultados de búsqueda textual
7. WHEN el Modo_Texto está activo, THEN THE Sistema_RAG SHALL asignar peso 0.60 a resultados de búsqueda por entidades

### Requirement 3: Generación de Respuesta para Texto Puro

**User Story:** Como usuario realizando una consulta sin imagen, quiero recibir una respuesta enciclopédica basada en el manual, para obtener información teórica precisa sobre histología.

#### Acceptance Criteria

1. WHEN el Modo_Texto está activo Y contexto_suficiente es verdadero, THEN THE Sistema_RAG SHALL utilizar system_prompt optimizado para consultas de texto
2. THE system_prompt para Modo_Texto SHALL instruir al LLM a responder basándose exclusivamente en el contenido del manual
3. THE system_prompt para Modo_Texto SHALL instruir al LLM a citar fuentes con formato "[Manual: archivo]"
4. THE system_prompt para Modo_Texto SHALL prohibir referencias a imágenes del usuario
5. WHEN el Modo_Texto está activo Y contexto_suficiente es falso, THEN THE Sistema_RAG SHALL generar mensaje amable indicando que no se encontró información
6. THE mensaje de contexto insuficiente en Modo_Texto SHALL incluir muestra del temario disponible
7. THE mensaje de contexto insuficiente en Modo_Texto SHALL sugerir reformular con términos más específicos o subir una imagen

### Requirement 4: Clasificación Semántica sin Imagen

**User Story:** Como usuario realizando una consulta teórica, quiero que el sistema valide si mi consulta pertenece al dominio de histología, para recibir retroalimentación apropiada si está fuera de alcance.

#### Acceptance Criteria

1. WHEN el Modo_Texto está activo, THEN THE ClasificadorSemantico SHALL calcular similitud coseno contra el temario extraído
2. WHEN el temario está vacío, THEN THE ClasificadorSemantico SHALL utilizar anclas semánticas hardcodeadas como fallback
3. THE ClasificadorSemantico SHALL aplicar umbral_efectivo calculado como UMBRAL_SIMILITUD multiplicado por 1.0 en Modo_Texto
4. WHEN la similitud supera umbral_efectivo, THEN THE ClasificadorSemantico SHALL retornar valido=True con metodo="semantico_imagebind"
5. WHEN la similitud no supera umbral_efectivo, THEN THE ClasificadorSemantico SHALL invocar LLM como árbitro
6. THE ClasificadorSemantico SHALL incluir muestra del temario en el prompt del LLM árbitro
7. WHEN el LLM árbitro retorna valido=False, THEN THE Sistema_RAG SHALL dirigir el flujo al nodo "fuera_temario"

### Requirement 5: Manejo de Contexto Insuficiente

**User Story:** Como usuario realizando una consulta de texto puro, quiero recibir un mensaje útil cuando no hay información disponible, para entender las limitaciones del sistema y cómo mejorar mi consulta.

#### Acceptance Criteria

1. WHEN el Modo_Texto está activo Y resultados_validos está vacío, THEN THE Sistema_RAG SHALL establecer contexto_suficiente como falso
2. WHEN contexto_suficiente es falso en Modo_Texto, THEN THE Sistema_RAG SHALL generar respuesta_final con formato de advertencia amable
3. THE respuesta_final SHALL incluir encabezado "No encontré información específica sobre eso en el manual"
4. THE respuesta_final SHALL incluir lista de hasta 15 temas disponibles del temario
5. THE respuesta_final SHALL incluir sugerencias: reformular con términos específicos, preguntar sobre temas listados, o subir imagen
6. THE Sistema_RAG SHALL registrar en trayectoria el evento con contexto_suficiente=False y modo="solo_texto"

### Requirement 6: Extracción de Términos de Búsqueda

**User Story:** Como sistema procesando una consulta de texto puro, necesito extraer términos histológicos relevantes de la consulta, para ejecutar búsquedas precisas en la base de datos.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL invocar LLM para extraer términos técnicos histológicos de consulta_texto
2. THE prompt de extracción SHALL solicitar categorías: TEJIDO, ESTRUCTURA, CONCEPTO, TINCIÓN, TÉRMINOS_CLAVE
3. WHEN el Modo_Texto está activo, THEN THE Sistema_RAG SHALL omitir análisis_visual del contexto de extracción
4. THE Sistema_RAG SHALL incluir contexto_memoria en el prompt de extracción si está disponible
5. THE Sistema_RAG SHALL almacenar resultado en state["terminos_busqueda"]
6. THE ExtractorEntidades SHALL extraer entidades estructuradas de consulta_texto para búsqueda por grafo
7. THE entidades_consulta SHALL contener listas de tejidos, estructuras y tinciones en minúsculas

### Requirement 7: Generación de Consultas de Búsqueda

**User Story:** Como sistema procesando una consulta de texto puro, necesito generar consultas optimizadas para búsqueda vectorial, para maximizar la relevancia de los resultados recuperados.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL invocar LLM para generar consulta_busqueda_texto corta (máximo 8 palabras)
2. WHEN el Modo_Texto está activo, THEN THE Sistema_RAG SHALL omitir generación de consulta_busqueda_visual
3. THE prompt de generación SHALL incluir terminos_busqueda y tema_encontrado si está disponible
4. THE Sistema_RAG SHALL almacenar resultado en state["consulta_busqueda_texto"]
5. IF la generación falla, THEN THE Sistema_RAG SHALL utilizar consulta_texto truncada a 77 caracteres como fallback
6. THE Sistema_RAG SHALL registrar en trayectoria la consulta generada

### Requirement 8: Integración con Memoria Semántica

**User Story:** Como usuario realizando múltiples consultas de texto, quiero que el sistema mantenga contexto de mis consultas anteriores, para recibir respuestas coherentes con el historial de conversación.

#### Acceptance Criteria

1. THE Memoria_Semantica SHALL mantener imagen_activa_path como None cuando no hay imagen
2. THE Memoria_Semantica SHALL incrementar turno_actual en cada interacción independientemente del modo
3. THE Memoria_Semantica SHALL almacenar query y response en conversations con imagen=None para Modo_Texto
4. WHEN turno_actual es múltiplo de 5, THEN THE Memoria_Semantica SHALL guardar resumen en Qdrant
5. THE resumen guardado en Qdrant SHALL incluir embedding_texto del resumen generado
6. WHEN no hay imagen_activa, THEN THE Memoria_Semantica SHALL utilizar vectores cero para embedding_uni y embedding_plip
7. THE Memoria_Semantica SHALL recuperar memorias históricas de Qdrant usando embedding de la consulta actual

### Requirement 9: Registro de Trayectoria

**User Story:** Como desarrollador del sistema, necesito registrar la trayectoria completa del procesamiento de consultas de texto puro, para depuración y análisis de rendimiento.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL registrar en trayectoria el nodo "Inicializar" con tiempo de ejecución
2. WHEN el Modo_Texto está activo, THEN THE trayectoria SHALL omitir el nodo "ProcesarImagen"
3. THE Sistema_RAG SHALL registrar en trayectoria el nodo "Clasificar" con tema_valido, tema_encontrado, entidades y similitud_dominio
4. THE Sistema_RAG SHALL registrar en trayectoria el nodo "GenerarConsulta" con query generada
5. THE Sistema_RAG SHALL registrar en trayectoria el nodo "BuscarNeo4j" con número de hits
6. THE Sistema_RAG SHALL registrar en trayectoria el nodo "FiltrarContexto" con hits_validos, imgs=0 y modo="solo_texto"
7. WHEN el Modo_Texto está activo, THEN THE trayectoria SHALL omitir el nodo "AnalisisComparativo"
8. THE Sistema_RAG SHALL registrar en trayectoria el nodo "GenerarRespuesta" con contexto_suficiente y modo="solo_texto"
9. THE Sistema_RAG SHALL guardar trayectoria completa en archivo trayectoria_neo4j.json

### Requirement 10: Validación de Resultados de Búsqueda

**User Story:** Como sistema procesando una consulta de texto puro, necesito validar que los resultados de búsqueda superen los umbrales apropiados, para garantizar calidad de las respuestas generadas.

#### Acceptance Criteria

1. THE Sistema_RAG SHALL iterar sobre resultados_busqueda para filtrar por umbral
2. WHEN resultado es tipo "texto" Y similitud es menor que umbral_texto, THEN THE Sistema_RAG SHALL rechazar el resultado
3. WHEN resultado es tipo "imagen" Y similitud es menor que umbral_imagen, THEN THE Sistema_RAG SHALL rechazar el resultado
4. WHEN resultado es tipo "imagen" Y imagen_path no existe en disco, THEN THE Sistema_RAG SHALL rechazar el resultado
5. THE Sistema_RAG SHALL almacenar resultados aprobados en resultados_validos
6. THE Sistema_RAG SHALL establecer contexto_suficiente como verdadero si resultados_validos no está vacío
7. THE Sistema_RAG SHALL construir contexto_documentos concatenando texto de resultados_validos ordenados por similitud

### Requirement 11: Formato de Respuesta Final

**User Story:** Como usuario realizando una consulta de texto puro, quiero recibir una respuesta bien estructurada con citas apropiadas, para entender la fuente de la información proporcionada.

#### Acceptance Criteria

1. THE respuesta_final en Modo_Texto SHALL incluir respuesta directa basada en el manual
2. THE respuesta_final SHALL incluir características histológicas relevantes según la base de datos
3. THE respuesta_final SHALL incluir citas con formato "[Manual: nombre_archivo]"
4. THE respuesta_final SHALL incluir conclusión al final
5. THE respuesta_final SHALL omitir cualquier referencia a imágenes del usuario
6. THE respuesta_final SHALL omitir sección de análisis comparativo
7. WHEN estructura_identificada está presente en Modo_Texto, THEN THE Sistema_RAG SHALL omitirla de la respuesta

### Requirement 12: Manejo de Errores en Modo Texto

**User Story:** Como sistema procesando una consulta de texto puro, necesito manejar errores de manera robusta, para proporcionar respuestas útiles incluso cuando ocurren fallos parciales.

#### Acceptance Criteria

1. WHEN la generación de embedding_texto falla, THEN THE Sistema_RAG SHALL establecer texto_embedding como None y continuar
2. WHEN la extracción de términos falla, THEN THE Sistema_RAG SHALL utilizar consulta_texto como terminos_busqueda
3. WHEN la búsqueda vectorial en Neo4j falla, THEN THE Sistema_RAG SHALL retornar lista vacía y registrar advertencia
4. WHEN la búsqueda por entidades falla, THEN THE Sistema_RAG SHALL retornar lista vacía y registrar advertencia
5. WHEN la invocación del LLM falla con error 429 o 503, THEN THE Sistema_RAG SHALL reintentar hasta 5 veces con espera exponencial
6. WHEN todos los reintentos fallan, THEN THE Sistema_RAG SHALL propagar la excepción
7. THE Sistema_RAG SHALL registrar todos los errores con nivel de advertencia en la consola
