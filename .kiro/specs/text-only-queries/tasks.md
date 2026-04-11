# Plan de Implementación: Text-Only Queries

## Descripción General

Este plan descompone la implementación de la funcionalidad de consultas de texto puro para el sistema RAG multimodal de histología. El sistema v4.2 ya tiene una implementación parcial con flujo bifurcado en LangGraph. Las tareas se enfocan en completar, refinar y validar según los requisitos y diseño formalizados.

## Tareas

- [x] 1. Validar y refinar router condicional `_route_por_modo`
  - Verificar que detecta correctamente modo texto cuando no hay `imagen_path` ni imagen en memoria
  - Asegurar que registra el modo detectado en trayectoria
  - Validar que el flujo omite `procesar_imagen` en modo texto
  - _Requisitos: 1.1, 1.3, 1.4_

- [ ]* 1.1 Escribir property test para router condicional
  - **Property 1: Determinismo del routing**
  - **Valida: Requisitos 1.1, 1.2**
  - Propiedad: Para cualquier estado con imagen_path válida O imagen en memoria → retorna "con_imagen"; caso contrario → "solo_texto"

- [x] 2. Implementar umbrales diferenciados en `_nodo_filtrar_contexto`
  - Modificar lógica para aplicar umbral 0.45 para texto en modo solo_texto
  - Mantener umbral 0.6 para texto en modo multimodal
  - Mantener umbral 0.45 para imágenes (constante)
  - Registrar modo utilizado en trayectoria
  - _Requisitos: 2.1, 2.2_

- [ ]* 2.1 Escribir tests unitarios para umbrales diferenciados
  - Test: modo texto con similitud 0.50 → resultado aceptado
  - Test: modo multimodal con similitud 0.50 → resultado rechazado
  - Test: modo multimodal con similitud 0.65 → resultado aceptado
  - _Requisitos: 2.1, 2.2_

- [x] 3. Implementar system prompts diferenciados en `_nodo_generar_respuesta`
  - Crear constante `SYSTEM_PROMPT_TEXTO` para modo texto puro
  - Crear constante `SYSTEM_PROMPT_MULTIMODAL` para modo con imagen
  - Modificar `_nodo_generar_respuesta` para seleccionar prompt según `state["tiene_imagen"]`
  - Asegurar que prompt de texto prohíbe referencias a imágenes del usuario
  - Asegurar que prompt de texto instruye citar con formato "[Manual: archivo]"
  - _Requisitos: 3.1, 3.2, 3.3, 3.4_

- [x] 4. Implementar manejo de contexto insuficiente para modo texto
  - Crear función `_generar_mensaje_sin_contexto_texto(temario: List[str]) -> str`
  - Modificar `_nodo_generar_respuesta` para usar mensaje amable cuando `contexto_suficiente=False` y `es_solo_texto=True`
  - Incluir muestra de hasta 15 temas del temario
  - Incluir sugerencias: reformular, preguntar sobre temas listados, subir imagen
  - _Requisitos: 3.5, 3.6, 3.7, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 4.1 Escribir tests unitarios para mensaje de contexto insuficiente
  - Test: mensaje incluye encabezado correcto
  - Test: mensaje incluye máximo 15 temas
  - Test: mensaje incluye las 3 sugerencias
  - _Requisitos: 5.3, 5.4, 5.5_

- [x] 5. Checkpoint - Validar flujo básico de texto puro
  - Ejecutar consulta de texto sin imagen
  - Verificar que omite `procesar_imagen` y `analisis_comparativo`
  - Verificar que aplica umbrales correctos
  - Verificar que genera respuesta con prompt de texto
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas

- [x] 6. Ajustar pesos de búsqueda híbrida en `Neo4jClient.busqueda_hibrida`
  - Verificar que en modo texto: peso_texto=0.80, peso_uni=0.20, peso_plip=0.20, peso_entidades=0.60
  - Verificar que en modo multimodal: peso_texto=0.40, peso_uni=0.70, peso_plip=0.70, peso_entidades=0.60
  - Peso de vecindad siempre 0.20
  - _Requisitos: 2.6, 2.7_

- [ ]* 6.1 Escribir property test para pesos de búsqueda
  - **Property 2: Dominancia de texto en modo solo_texto**
  - **Valida: Requisitos 2.6**
  - Propiedad: En modo texto, resultados textuales tienen mayor peso acumulado que visuales

- [x] 7. Validar integración con `SemanticMemory` para modo texto
  - Verificar que `add_interaction` guarda interacciones con `imagen=None` en modo texto
  - Verificar que `_guardar_memoria_qdrant` usa vectores cero para `embedding_uni` y `embedding_plip` cuando no hay imagen
  - Verificar que `get_context` recupera memorias históricas correctamente en modo texto
  - _Requisitos: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [ ]* 7.1 Escribir tests unitarios para memoria en modo texto
  - Test: interacción sin imagen se guarda con `imagen=None`
  - Test: resumen en Qdrant usa vectores cero para embeddings visuales
  - Test: recuperación de contexto funciona sin imagen activa
  - _Requisitos: 8.1, 8.6_

- [x] 8. Completar registro de trayectoria para modo texto
  - Verificar que `_nodo_inicializar` registra tiempo de ejecución
  - Verificar que trayectoria omite `ProcesarImagen` en modo texto
  - Verificar que `_nodo_clasificar` registra entidades, similitud_dominio y método
  - Verificar que `_nodo_filtrar_contexto` registra modo="solo_texto" e imgs=0
  - Verificar que trayectoria omite `AnalisisComparativo` en modo texto
  - Verificar que `_nodo_generar_respuesta` registra contexto_suficiente y modo
  - Verificar que trayectoria se guarda en `trayectoria_neo4j.json`
  - _Requisitos: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9_

- [ ]* 8.1 Escribir property test para trayectoria
  - **Property 3: Completitud de trayectoria**
  - **Valida: Requisitos 9.1-9.9**
  - Propiedad: Toda ejecución en modo texto debe registrar nodos: Inicializar, Clasificar, GenerarConsulta, BuscarNeo4j, FiltrarContexto, GenerarRespuesta, Finalizar (sin ProcesarImagen ni AnalisisComparativo)

- [x] 9. Validar clasificador semántico en modo texto
  - Verificar que `ClasificadorSemantico.clasificar` calcula similitud contra temario
  - Verificar que usa anclas semánticas como fallback si temario está vacío
  - Verificar que aplica umbral_efectivo sin ajuste en modo texto (multiplicador 1.0)
  - Verificar que invoca LLM como árbitro cuando similitud < umbral
  - Verificar que incluye muestra del temario en prompt del LLM
  - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 10. Validar extracción de términos en modo texto
  - Verificar que `_nodo_clasificar` invoca LLM para extraer términos histológicos
  - Verificar que omite `analisis_visual` del contexto cuando no hay imagen
  - Verificar que incluye `contexto_memoria` si está disponible
  - Verificar que `ExtractorEntidades.extraer_de_texto` extrae tejidos, estructuras y tinciones
  - _Requisitos: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 11. Validar generación de consultas de búsqueda en modo texto
  - Verificar que `_nodo_generar_consulta` genera `consulta_busqueda_texto` (máximo 8 palabras)
  - Verificar que omite generación de `consulta_busqueda_visual` en modo texto
  - Verificar que incluye `terminos_busqueda` y `tema_encontrado` en el prompt
  - Verificar que usa consulta_texto truncada como fallback si falla la generación
  - Verificar que registra consulta generada en trayectoria
  - _Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 12. Validar validación de resultados de búsqueda
  - Verificar que `_nodo_filtrar_contexto` itera sobre `resultados_busqueda`
  - Verificar que rechaza resultados de texto con similitud < umbral_texto
  - Verificar que rechaza resultados de imagen con similitud < umbral_imagen
  - Verificar que rechaza imágenes cuyo path no existe en disco
  - Verificar que almacena resultados aprobados en `resultados_validos`
  - Verificar que establece `contexto_suficiente=True` si hay resultados válidos
  - _Requisitos: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 13. Validar formato de respuesta final en modo texto
  - Verificar que respuesta incluye contenido basado en el manual
  - Verificar que incluye citas con formato "[Manual: nombre_archivo]"
  - Verificar que omite referencias a imágenes del usuario
  - Verificar que omite sección de análisis comparativo
  - Verificar que omite `estructura_identificada` en modo texto
  - _Requisitos: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

- [x] 14. Implementar manejo robusto de errores en modo texto
  - Verificar que fallo en `embed_query_con_reintento` establece `texto_embedding=None` y continúa
  - Verificar que fallo en extracción de términos usa `consulta_texto` como fallback
  - Verificar que fallo en búsqueda vectorial retorna lista vacía y registra advertencia
  - Verificar que fallo en búsqueda por entidades retorna lista vacía y registra advertencia
  - Verificar que fallo en invocación LLM reintenta hasta 5 veces con espera exponencial
  - Verificar que todos los errores se registran con nivel de advertencia
  - _Requisitos: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

- [ ]* 14.1 Escribir tests unitarios para manejo de errores
  - Test: fallo en embedding no detiene el flujo
  - Test: fallo en extracción usa fallback
  - Test: fallo en búsqueda retorna lista vacía
  - _Requisitos: 12.1, 12.2, 12.3, 12.4_

- [x] 15. Checkpoint final - Validación end-to-end
  - Ejecutar suite completa de consultas de texto puro
  - Verificar transiciones fluidas entre consultas con y sin imagen
  - Verificar que memoria mantiene coherencia entre modos
  - Verificar que trayectoria se registra correctamente en todos los casos
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Los property tests validan propiedades universales de corrección
- Los tests unitarios validan ejemplos específicos y casos borde
- El sistema v4.2 ya tiene la estructura base implementada, por lo que las tareas se enfocan en validación, refinamiento y completitud
