from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any, Optional

from utils.config import (
    INDEX_TEXTO, INDEX_UNI, INDEX_PLIP, DIM_TEXTO, DIM_IMG_UNI, DIM_IMG_PLIP, 
    SIMILAR_IMG_THRESHOLD, NEO4J_GRAPH_DEPTH
)

class Neo4jClient:
    """
    Cliente asíncrono para gestionar la conexión y operaciones sobre la base de datos Neo4j.
    
    Esta clase encapsula toda la interacción con el grafo de Neo4j, incluyendo:
    - Conexión al servidor (local o AuraDB).
    - Creación y validación del esquema (constraints e índices vectoriales).
    - Operaciones de escritura (upsert de chunks, PDFs, imágenes).
    - Creación de relaciones de similitud visual.
    - Operaciones de lectura complejas (búsqueda vectorial, por entidades, expansión de vecindad y búsqueda híbrida).
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Inicializa el cliente de Neo4j.

        Args:
            uri (str): URI de conexión a Neo4j (ej. bolt://localhost:7687).
            user (str): Usuario de autenticación.
            password (str): Contraseña.
        """
        self.uri      = uri
        self.user     = user
        self.password = password
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        """
        Establece y verifica la conexión al driver asíncrono de Neo4j.
        """
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        await self._driver.verify_connectivity()
        print(f"✅ Neo4j conectado: {self.uri}")

    async def close(self):
        """
        Cierra la conexión activa al driver de Neo4j.
        """
        if self._driver:
            await self._driver.close()

    async def run(self, query: str, params: Dict = None) -> List[Dict]:
        """
        Ejecuta una consulta Cypher de forma asíncrona.

        Args:
            query (str): Consulta Cypher a ejecutar.
            params (Dict, opcional): Diccionario de parámetros para la consulta.

        Returns:
            List[Dict]: Lista de resultados convertidos a diccionarios.
        """
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) for record in await result.data()]

    async def crear_esquema(self):
        """
        Configura el esquema inicial de la base de datos Neo4j.
        Crea las restricciones de unicidad (constraints) necesarias y los índices vectoriales
        para texto, imágenes de UNI y PLIP. Si las dimensiones del índice vectorial 
        existente no coinciden con las definidas, el índice se elimina y recrea.
        """
        print("🏗️ Creando esquema Neo4j (v4.2 UNI + PLIP)...")
        constraints = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT imagen_id IF NOT EXISTS FOR (i:Imagen) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT pdf_nombre IF NOT EXISTS FOR (p:PDF) REQUIRE p.nombre IS UNIQUE",
            "CREATE CONSTRAINT tejido_nombre IF NOT EXISTS FOR (t:Tejido) REQUIRE t.nombre IS UNIQUE",
            "CREATE CONSTRAINT estructura_nombre IF NOT EXISTS FOR (e:Estructura) REQUIRE e.nombre IS UNIQUE",
            "CREATE CONSTRAINT tincion_nombre IF NOT EXISTS FOR (t:Tincion) REQUIRE t.nombre IS UNIQUE",
        ]
        for c in constraints:
            try:
                await self.run(c)
            except Exception as e:
                print(f"  ⚠️ Constraint: {e}")

        # --- Validación de Índices Vectoriales (Dimensionalidad) ---
        try:
            indexes = await self.run("SHOW INDEXES YIELD name, type, options")
            for idx in indexes:
                if idx["type"] == "VECTOR":
                    name = idx["name"]
                    config = idx["options"].get("indexConfig", {})
                    dims = config.get("vector.dimensions")
                    
                    target_dims = None
                    if name == INDEX_TEXTO: target_dims = DIM_TEXTO
                    elif name == INDEX_UNI: target_dims = DIM_IMG_UNI
                    elif name == INDEX_PLIP: target_dims = DIM_IMG_PLIP
                    
                    if target_dims and dims != target_dims:
                        print(f"  ⚠️ Dimensión incorrecta en índice Neo4j '{name}' ({dims} != {target_dims}). Recreando...")
                        await self.run(f"DROP INDEX {name}")
        except Exception as e:
            print(f"  ⚠️ Error validando índices Neo4j: {e}")

        # 3 Índices Vectoriales
        vector_queries = [
            # 1. TEXTO
            f"""
            CREATE VECTOR INDEX {INDEX_TEXTO} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {DIM_TEXTO},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            # 2. IMAGEN UNI
            f"""
            CREATE VECTOR INDEX {INDEX_UNI} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_uni
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {DIM_IMG_UNI},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            # 3. IMAGEN PLIP
            f"""
            CREATE VECTOR INDEX {INDEX_PLIP} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_plip
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {DIM_IMG_PLIP},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        for vq in vector_queries:
            try:
                await self.run(vq)
            except Exception as e:
                print(f"  ⚠️ Vector index error: {e}")

        print("✅ Esquema Neo4j listo (3 índices)")

    async def upsert_pdf(self, nombre: str):
        """
        Actualiza o inserta (Upsert) un nodo PDF en el grafo.
        
        Args:
            nombre (str): Nombre del archivo PDF.
        """
        await self.run("MERGE (p:PDF {nombre: $nombre})", {"nombre": nombre})

    async def upsert_chunk(self, chunk_id: str, texto: str, fuente: str,
                            chunk_idx: int, embedding: List[float],
                            entidades: Dict[str, List[str]]):
        """
        Actualiza o inserta un chunk de texto junto con sus entidades asociadas.
        Crea las relaciones entre el chunk y el PDF correspondiente, así como 
        las relaciones `MENCIONA` con Tejidos, Estructuras y Tinciones. Adicionalmente, 
        infiere relaciones entre las entidades (ej. Tejido CONTIENE Estructura).

        Args:
            chunk_id (str): Identificador único del chunk.
            texto (str): Contenido del chunk de texto.
            fuente (str): Nombre del PDF origen.
            chunk_idx (int): Índice de orden del chunk en el texto fuente.
            embedding (List[float]): Vector de embedding del texto.
            entidades (Dict[str, List[str]]): Entidades divididas en 'tejidos', 'estructuras' y 'tinciones'.
        """
        await self.run("""
            MERGE (c:Chunk {id: $id})
            SET c.texto = $texto, c.fuente = $fuente,
                c.chunk_id = $chunk_idx, c.embedding = $embedding
            WITH c
            MERGE (pdf:PDF {nombre: $fuente})
            MERGE (c)-[:PERTENECE_A]->(pdf)
        """, {
            "id": chunk_id, "texto": texto, "fuente": fuente,
            "chunk_idx": chunk_idx, "embedding": embedding
        })
        for tejido in entidades.get("tejidos", []):
            await self.run("""
                MERGE (t:Tejido {nombre: $nombre})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"nombre": tejido, "chunk_id": chunk_id})
        for estructura in entidades.get("estructuras", []):
            await self.run("""
                MERGE (e:Estructura {nombre: $nombre})
                WITH e MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
            """, {"nombre": estructura, "chunk_id": chunk_id})
        for tincion in entidades.get("tinciones", []):
            await self.run("""
                MERGE (t:Tincion {nombre: $nombre})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"nombre": tincion, "chunk_id": chunk_id})
        for tejido in entidades.get("tejidos", []):
            for estructura in entidades.get("estructuras", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tejido})
                    MERGE (e:Estructura {nombre: $estructura})
                    MERGE (t)-[:CONTIENE]->(e)
                """, {"tejido": tejido, "estructura": estructura})
            for tincion in entidades.get("tinciones", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tejido})
                    MERGE (ti:Tincion {nombre: $tincion})
                    MERGE (t)-[:TENIDA_CON]->(ti)
                """, {"tejido": tejido, "tincion": tincion})
        for estructura in entidades.get("estructuras", []):
            for tincion in entidades.get("tinciones", []):
                await self.run("""
                    MERGE (e:Estructura {nombre: $estructura})
                    MERGE (ti:Tincion {nombre: $tincion})
                    MERGE (e)-[:TENIDA_CON]->(ti)
                """, {"estructura": estructura, "tincion": tincion})

    async def upsert_imagen(self, imagen_id: str, path: str, fuente: str,
                             pagina: int, ocr_text: str, texto_pagina: str,
                             emb_uni: List[float], emb_plip: List[float]):
        """
        Actualiza o inserta datos de una imagen procesada en Neo4j.
        Adjunta embeddings multimodales (UNI y PLIP), metadatos (página, texto OCR), y establece
        relaciones de pertenencia al PDF iterando por página.

        Args:
            imagen_id (str): Identificador único de la imagen.
            path (str): Ruta local del archivo de imagen.
            fuente (str): Nombre del documento fuente (PDF).
            pagina (int): Número de página donde se encontró la imagen.
            ocr_text (str): Texto extraído directamente de la imagen mediante OCR.
            texto_pagina (str): Texto de la página circundante (para contexto).
            emb_uni (List[float]): Vector de embedding visual UNI.
            emb_plip (List[float]): Vector de embedding visual PLIP.
        """
        await self.run("""
            MERGE (i:Imagen {id: $id})
            SET i.path = $path, i.fuente = $fuente,
                i.pagina = $pagina, i.ocr_text = $ocr_text,
                i.texto_pagina = $texto_pagina,
                i.embedding_uni = $emb_uni,
                i.embedding_plip = $emb_plip
            WITH i
            MERGE (pdf:PDF {nombre: $fuente})
            MERGE (i)-[:PERTENECE_A]->(pdf)
            MERGE (pag:Pagina {numero: $pagina, pdf_nombre: $fuente})
            MERGE (i)-[:EN_PAGINA]->(pag)
        """, {
            "id": imagen_id, "path": path, "fuente": fuente,
            "pagina": pagina, "ocr_text": ocr_text, "texto_pagina": texto_pagina,
            "emb_uni": emb_uni, "emb_plip": emb_plip
        })

    async def crear_relaciones_similitud(self, umbral: float = SIMILAR_IMG_THRESHOLD):
        """
        Construye relaciones explícitas ':SIMILAR_A' en el grafo para imágenes,
        utilizando sus vectores UNI para evaluar distancia coseno y conectándolas
        si superan el umbral definido.

        Args:
            umbral (float, opcional): Umbral de similitud para crear la conexión. Default configurado globalmente.
        """
        print(f"🔗 Creando relaciones :SIMILAR_A (usando UNI, umbral={umbral})...")
        imagenes = await self.run(
            "MATCH (i:Imagen) WHERE i.embedding_uni IS NOT NULL RETURN i.id AS id, i.embedding_uni AS emb"
        )
        if len(imagenes) < 2:
            return
        creadas = 0
        for img in imagenes:
            resultado = await self.run("""
                CALL db.index.vector.queryNodes($index, 6, $emb)
                YIELD node AS vecino, score
                WHERE vecino.id <> $id AND score >= $umbral
                WITH vecino, score
                MATCH (origen:Imagen {id: $id})
                MERGE (origen)-[r:SIMILAR_A]->(vecino)
                SET r.score = score
                RETURN count(*) AS n
            """, {
                "index": INDEX_UNI,
                "emb": img["emb"], "id": img["id"], "umbral": umbral
            })
            creadas += resultado[0]["n"] if resultado else 0
        print(f"✅ {creadas} relaciones :SIMILAR_A creadas")

    async def busqueda_vectorial(self, embedding: List[float],
                                  index_name: str, top_k: int = 10) -> List[Dict]:
        """
        Búsqueda general en base a similitud coseno en un índice vectorial específico.
        Las consultas divergen dependiendo de si el índice apuntado guarda texto ('Chunk')
        o representaciones visuales ('Imagen').

        Args:
            embedding (List[float]): Vector a buscar en el índice.
            index_name (str): Nombre del índice Neo4j.
            top_k (int, opcional): Número máximo de nodos de resultado. Por defecto 10.

        Returns:
            List[Dict]: Resultados con campos estandarizados (id, texto, fuente, tipo, imagen_path, similitud).
        """
        is_text_index = index_name == INDEX_TEXTO
        if is_text_index:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS c, score
                RETURN c.id AS id, c.texto AS texto, c.fuente AS fuente,
                       'texto' AS tipo, null AS imagen_path, score AS similitud
                ORDER BY similitud DESC
            """
        else:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS i, score
                RETURN i.id AS id, 
                       coalesce(i.texto_pagina, i.ocr_text) AS texto, 
                       i.fuente AS fuente,
                       'imagen' AS tipo, i.path AS imagen_path, score AS similitud
                ORDER BY similitud DESC
            """
        try:
            return await self.run(query, {"index": index_name, "emb": embedding, "k": top_k})
        except Exception as e:
            print(f"⚠️ Error búsqueda vectorial {index_name}: {e}")
            return []

    async def busqueda_por_entidades(self, entidades: Dict[str, List[str]],
                                      top_k: int = 10) -> List[Dict]:
        """
        Búsqueda por coincidencias semánticas del grafo basada en las entidades presentes
        en la consulta (Tejido, Estructura, Tinción). Busca chunks que estén relacionados.

        Args:
            entidades (Dict[str, List[str]]): Entidades identificadas (tejidos, estructuras, tinciones).
            top_k (int, opcional): Límite de resultados. Por defecto 10.

        Returns:
            List[Dict]: Chunks del manual que referencian o contienen alguna de las entidades.
        """
        tejidos     = entidades.get("tejidos", [])
        estructuras = entidades.get("estructuras", [])
        tinciones   = entidades.get("tinciones", [])
        if not any([tejidos, estructuras, tinciones]):
            return []

        where_clauses = []
        params: Dict[str, Any] = {}
        if tejidos:
            where_clauses.append("ANY(t IN tejidos WHERE t.nombre IN $tejidos)")
            params["tejidos"] = tejidos
        if estructuras:
            where_clauses.append("ANY(e IN estructuras WHERE e.nombre IN $estructuras)")
            params["estructuras"] = estructuras
        if tinciones:
            where_clauses.append("ANY(ti IN tinciones WHERE ti.nombre IN $tinciones)")
            params["tinciones"] = tinciones

        where_str = " OR ".join(where_clauses)
        query = f"""
            MATCH (c:Chunk)
            OPTIONAL MATCH (c)-[:MENCIONA]->(t:Tejido)
            OPTIONAL MATCH (c)-[:MENCIONA]->(e:Estructura)
            OPTIONAL MATCH (c)-[:MENCIONA]->(ti:Tincion)
            WITH c,
                 collect(DISTINCT t) AS tejidos,
                 collect(DISTINCT e) AS estructuras,
                 collect(DISTINCT ti) AS tinciones
            WHERE {where_str}
            RETURN c.id AS id, c.texto AS texto, c.fuente AS fuente,
                   'texto' AS tipo, null AS imagen_path, 0.49 AS similitud
            LIMIT $top_k
        """
        params["top_k"] = top_k
        try:
            return await self.run(query, params)
        except Exception as e:
            print(f"⚠️ Error búsqueda entidades: {e}")
            return []

    async def expandir_vecindad(self, node_ids: List[str],
                                 depth: int = NEO4J_GRAPH_DEPTH) -> List[Dict]:
        """
        Recorre el grafo alrededor de los nodos de alta relevancia para extraer el contexto
        subyacente en relaciones semánticas u organizacionales (vecindad algorítmica).
        Expande por pertenencia a PDF, entidades compartidas o similitud visual explícita.

        Args:
            node_ids (List[str]): IDs de ancla de donde se expande (los top_k resultados anteriores).
            depth (int, opcional): Profundidad de inferencia conceptual (no lineal).

        Returns:
            List[Dict]: Entidades conectadas extraídas formatadas estándar, con puntaje heuirístico degradado (0.95 fuerte o 0.3 periférico).
        """
        if not node_ids:
            return []
        query = """
            UNWIND $ids AS nid
            MATCH (n {id: nid})

            // Expansión 1: otros nodos en el mismo PDF (excluir el nodo origen)
            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)
            WHERE vecino_pdf.id <> nid
            WITH n, nid, collect(DISTINCT vecino_pdf)[..5] AS list_pdf

            // Expansión 2: chunks que comparten entidades (excluir el nodo origen)
            OPTIONAL MATCH (n)-[:MENCIONA]->(entidad)<-[:MENCIONA]-(vecino_entidad:Chunk)
            WHERE vecino_entidad.id <> nid
            WITH n, nid, list_pdf, collect(DISTINCT vecino_entidad)[..5] AS list_ent

            // Expansión 3: imágenes similares por embedding
            OPTIONAL MATCH (n)-[:SIMILAR_A]-(vecino_similar:Imagen)
            WITH n, nid, list_pdf, list_ent, collect(DISTINCT vecino_similar)[..5] AS list_sim

            // Expansión 4: imágenes de la misma página que el chunk
            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf2:PDF)<-[:PERTENECE_A]-(img_pag:Imagen)
            WITH n, nid, list_pdf, list_ent, list_sim, collect(DISTINCT img_pag)[..5] AS list_pag

            WITH n, $ids AS ids_originales,
                 list_pdf + list_ent + list_sim + list_pag AS vecinos_raw

            UNWIND vecinos_raw AS v

            WITH n, v, ids_originales
            WHERE v IS NOT NULL AND NOT v.id IN ids_originales

            RETURN DISTINCT
                v.id AS id,
                CASE 
                    WHEN v:Imagen THEN coalesce(v.texto_pagina, v.ocr_text, '') 
                    ELSE coalesce(v.texto, '') 
                END AS texto,
                v.fuente AS fuente,
                CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS tipo,
                CASE WHEN v:Imagen THEN v.path ELSE null END AS imagen_path,
                CASE 
                    WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
                         (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
                    THEN 0.95 
                    ELSE 0.3 
                END AS similitud
            LIMIT 15
        """
        try:
            return await self.run(query, {"ids": node_ids})
        except Exception as e:
            print(f"⚠️ Error expansión vecindad: {e}")
            return []

    async def busqueda_camino_semantico(self,
                                         tejido_origen: Optional[str],
                                         tejido_destino: Optional[str]) -> List[Dict]:
        """
        Intenta trazar una ruta algorítmica corta entre dos etiquetas de Tejido si la pregunta contiene
        una comparativa o relación entre ellos (distancia semántica en el manual).

        Args:
            tejido_origen (Optional[str]): Denominación del primer tejido.
            tejido_destino (Optional[str]): Denominación del tejido al que se traza ruta.

        Returns:
            List[Dict]: Chunks informativos hallados en el camino directo.
        """
        if not tejido_origen or not tejido_destino:
            return []
        query = """
            MATCH (origen {nombre: $origen}), (destino {nombre: $destino})
            MATCH path = shortestPath((origen)-[*1..4]-(destino))
            UNWIND nodes(path) AS nodo
            OPTIONAL MATCH (c:Chunk)-[:MENCIONA]->(nodo)
            RETURN DISTINCT c.id AS id, c.texto AS texto, c.fuente AS fuente,
                   'texto' AS tipo, null AS imagen_path, 0.4 AS similitud
            LIMIT 5
        """
        try:
            return await self.run(query, {"origen": tejido_origen, "destino": tejido_destino})
        except Exception as e:
            print(f"⚠️ Error búsqueda camino: {e}")
            return []

    async def busqueda_hibrida(self,
                                texto_embedding: Optional[List[float]],
                                imagen_embedding_uni: Optional[List[float]],
                                imagen_embedding_plip: Optional[List[float]],
                                entidades: Dict[str, List[str]],
                                top_k: int = 10) -> List[Dict]:
        """
        Orquesta transversalmente métodos de búsqueda para devolver la mezcla más relevante de resultados
        usando un sistema de suma ponderada. 
        Maneja vectores (Texto, UNI, PLIP), entidades exactas de Cypher y expansión automática del vecindario.

        Args:
            texto_embedding (Optional[List[float]]): Vector codificado de la consulta en sí.
            imagen_embedding_uni (Optional[List[float]]): Vector visual extraído por UNI (opcional si hay imagen).
            imagen_embedding_plip (Optional[List[float]]): Vector visual extraído por PLIP (opcional si hay imagen).
            entidades (Dict[str, List[str]]): Entidades aisladas previamente (tejidos, tinciones, estructuras).
            top_k (int, opcional): Solicitudes top base a índices (el return real puede superarlo al expandir y consolidar). Por defecto 10.

        Returns:
            List[Dict]: Los 15 mejores chunks integrados.
        """
        res_texto = []
        res_uni   = []
        res_plip  = []
        res_ent   = []
        res_vec   = []

        # 1. Búsqueda Texto (Gemini/Local)
        if texto_embedding is not None:
            res_texto = await self.busqueda_vectorial(texto_embedding, INDEX_TEXTO, top_k)

        # 2. Búsqueda Imagen UNI
        if imagen_embedding_uni is not None:
            res_uni = await self.busqueda_vectorial(imagen_embedding_uni, INDEX_UNI, top_k)

        # 3. Búsqueda Imagen PLIP
        if imagen_embedding_plip is not None:
            res_plip = await self.busqueda_vectorial(imagen_embedding_plip, INDEX_PLIP, top_k)

        # 4. Entidades
        res_ent = await self.busqueda_por_entidades(entidades, top_k)

        # Vecindad sobre los mejores
        todos = res_texto + res_uni + res_plip
        top_ids = [r["id"] for r in todos[:6] if r.get("id")]
        if top_ids:
            res_vec = await self.expandir_vecindad(top_ids)

        combined: Dict[str, Dict] = {}

        def agregar(resultados: List[Dict], peso: float):
            for r in resultados:
                key = r.get("id") or f"{r.get('fuente')}_{str(r.get('texto',''))[:40]}"
                if not r.get("texto") and not r.get("imagen_path"):
                    continue
                sim_ponderada = r.get("similitud", 0) * peso
                if key not in combined:
                    combined[key] = {**r, "similitud": sim_ponderada}
                else:
                    combined[key]["similitud"] += sim_ponderada

        # Pesos ajustados
        agregar(res_texto, 0.80) # Texto sigue siendo fundamental
        agregar(res_uni,   0.50) # UNI complemento visual
        agregar(res_plip,  0.50) # PLIP complemento visual
        agregar(res_ent,   0.60) # Entidades (¡Crucial para discriminar órganos!)
        agregar(res_vec,   0.20)

        final = sorted(combined.values(), key=lambda x: x["similitud"], reverse=True)

        print(f"   📊 Híbrida: Txt={len(res_texto)} | "
              f"UNI={len(res_uni)} | PLIP={len(res_plip)} | Ent={len(res_ent)} | Vec={len(res_vec)} -> {len(final)}")

        return final[:15]
