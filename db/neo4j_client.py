from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any, Optional

from utils.config import (
    TEXT_INDEX_NAME, UNI_INDEX_NAME, PLIP_INDEX_NAME, TEXT_DIM, UNI_IMG_DIM, PLIP_IMG_DIM, 
    SIMILAR_IMG_THRESHOLD, NEO4J_GRAPH_DEPTH
)

class Neo4jClient:
    """
    Asynchronous client for managing Neo4j connection and graph operations.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri      = uri
        self.user     = user
        self.password = password
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        await self._driver.verify_connectivity()
        print(f"✅ Neo4j Connected: {self.uri}")

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def run(self, query: str, params: Dict = None) -> List[Dict]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) for record in await result.data()]

    async def create_schema(self):
        print("🏗️ Creating Neo4j Schema (v4.4 UNI + PLIP)...")
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

        try:
            indexes = await self.run("SHOW INDEXES YIELD name, type, options")
            for idx in indexes:
                if idx["type"] == "VECTOR":
                    name = idx["name"]
                    config = idx["options"].get("indexConfig", {})
                    dims = config.get("vector.dimensions")
                    
                    target_dims = None
                    if name == TEXT_INDEX_NAME: target_dims = TEXT_DIM
                    elif name == UNI_INDEX_NAME: target_dims = UNI_IMG_DIM
                    elif name == PLIP_INDEX_NAME: target_dims = PLIP_IMG_DIM
                    
                    if target_dims and dims != target_dims:
                        print(f"  ⚠️ Incorrect dimension on index '{name}' ({dims} != {target_dims}). Recreating...")
                        await self.run(f"DROP INDEX {name}")
        except Exception as e:
            print(f"  ⚠️ Schema validation error: {e}")

        vector_queries = [
            f"""
            CREATE VECTOR INDEX {TEXT_INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {TEXT_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX {UNI_INDEX_NAME} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_uni
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {UNI_IMG_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            f"""
            CREATE VECTOR INDEX {PLIP_INDEX_NAME} IF NOT EXISTS
            FOR (i:Imagen) ON i.embedding_plip
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {PLIP_IMG_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
        ]
        for vq in vector_queries:
            try:
                await self.run(vq)
            except Exception as e:
                print(f"  ⚠️ Vector index error: {e}")

        print("✅ Neo4j Schema Ready (3 indexes)")

    async def upsert_pdf(self, name: str):
        await self.run("MERGE (p:PDF {nombre: $name})", {"name": name})

    async def upsert_chunk(self, chunk_id: str, text: str, source: str,
                            chunk_idx: int, embedding: List[float],
                            entities: Dict[str, List[str]]):
        await self.run("""
            MERGE (c:Chunk {id: $id})
            SET c.texto = $text, c.fuente = $source,
                c.chunk_id = $chunk_idx, c.embedding = $embedding
            WITH c
            MERGE (pdf:PDF {nombre: $source})
            MERGE (c)-[:PERTENECE_A]->(pdf)
        """, {
            "id": chunk_id, "text": text, "source": source,
            "chunk_idx": chunk_idx, "embedding": embedding
        })
        for tissue in entities.get("tejidos", []):
            await self.run("""
                MERGE (t:Tejido {nombre: $name})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"name": tissue, "chunk_id": chunk_id})
        for structure in entities.get("estructuras", []):
            await self.run("""
                MERGE (e:Estructura {nombre: $name})
                WITH e MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(e)
            """, {"name": structure, "chunk_id": chunk_id})
        for stain in entities.get("tinciones", []):
            await self.run("""
                MERGE (t:Tincion {nombre: $name})
                WITH t MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENCIONA]->(t)
            """, {"name": stain, "chunk_id": chunk_id})
        for tissue in entities.get("tejidos", []):
            for structure in entities.get("estructuras", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tissue})
                    MERGE (e:Estructura {nombre: $structure})
                    MERGE (t)-[:CONTIENE]->(e)
                """, {"tissue": tissue, "structure": structure})
            for stain in entities.get("tinciones", []):
                await self.run("""
                    MERGE (t:Tejido {nombre: $tissue})
                    MERGE (ti:Tincion {nombre: $stain})
                    MERGE (t)-[:TENIDA_CON]->(ti)
                """, {"tissue": tissue, "stain": stain})
        for structure in entities.get("estructuras", []):
            for stain in entities.get("tinciones", []):
                await self.run("""
                    MERGE (e:Estructura {nombre: $structure})
                    MERGE (ti:Tincion {nombre: $stain})
                    MERGE (e)-[:TENIDA_CON]->(ti)
                """, {"structure": structure, "stain": stain})

    async def upsert_image(self, image_id: str, path: str, source: str,
                             page: int, ocr_text: str, page_text: str,
                             emb_uni: List[float], emb_plip: List[float]):
        await self.run("""
            MERGE (i:Imagen {id: $id})
            SET i.path = $path, i.fuente = $source,
                i.pagina = $page, i.ocr_text = $ocr_text,
                i.texto_pagina = $page_text,
                i.embedding_uni = $emb_uni,
                i.embedding_plip = $emb_plip
            WITH i
            MERGE (pdf:PDF {nombre: $source})
            MERGE (i)-[:PERTENECE_A]->(pdf)
            MERGE (pag:Pagina {numero: $page, pdf_nombre: $source})
            MERGE (i)-[:EN_PAGINA]->(pag)
        """, {
            "id": image_id, "path": path, "source": source,
            "page": page, "ocr_text": ocr_text, "page_text": page_text,
            "emb_uni": emb_uni, "emb_plip": emb_plip
        })

    async def create_similarity_relations(self, threshold: float = SIMILAR_IMG_THRESHOLD):
        print(f"🔗 Creating :SIMILAR_A relations (UNI method, threshold={threshold})...")
        images = await self.run(
            "MATCH (i:Imagen) WHERE i.embedding_uni IS NOT NULL RETURN i.id AS id, i.embedding_uni AS emb"
        )
        if len(images) < 2:
            return
        created = 0
        for img in images:
            result = await self.run("""
                CALL db.index.vector.queryNodes($index, 6, $emb)
                YIELD node AS vecino, score
                WHERE vecino.id <> $id AND score >= $threshold
                WITH vecino, score
                MATCH (origen:Imagen {id: $id})
                MERGE (origen)-[r:SIMILAR_A]->(vecino)
                SET r.score = score
                RETURN count(*) AS n
            """, {
                "index": UNI_INDEX_NAME,
                "emb": img["emb"], "id": img["id"], "threshold": threshold
            })
            created += result[0]["n"] if result else 0
        print(f"✅ {created} :SIMILAR_A relations linked")

    async def vector_search(self, embedding: List[float],
                                  index_name: str, top_k: int = 10) -> List[Dict]:
        is_text_index = index_name == TEXT_INDEX_NAME
        if is_text_index:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS c, score
                RETURN c.id AS id, c.texto AS text, c.fuente AS source,
                       'texto' AS type, null AS image_path, score AS similarity
                ORDER BY similarity DESC
            """
        else:
             query = """
                CALL db.index.vector.queryNodes($index, $k, $emb)
                YIELD node AS i, score
                RETURN i.id AS id, 
                       coalesce(i.texto_pagina, i.ocr_text) AS text, 
                       i.fuente AS source,
                       'imagen' AS type, i.path AS image_path, score AS similarity
                ORDER BY similarity DESC
            """
        try:
            return await self.run(query, {"index": index_name, "emb": embedding, "k": top_k})
        except Exception as e:
            print(f"⚠️ Vector search error {index_name}: {e}")
            return []

    async def entity_search(self, entities: Dict[str, List[str]],
                                      top_k: int = 10) -> List[Dict]:
        tissues    = entities.get("tejidos", [])
        structures = entities.get("estructuras", [])
        stains     = entities.get("tinciones", [])
        if not any([tissues, structures, stains]):
            return []

        where_clauses = []
        params: Dict[str, Any] = {}
        if tissues:
            where_clauses.append("ANY(t IN tejidos WHERE t.nombre IN $tejidos)")
            params["tejidos"] = tissues
        if structures:
            where_clauses.append("ANY(e IN estructuras WHERE e.nombre IN $estructuras)")
            params["estructuras"] = structures
        if stains:
            where_clauses.append("ANY(ti IN tinciones WHERE ti.nombre IN $tinciones)")
            params["tinciones"] = stains

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
            RETURN c.id AS id, c.texto AS text, c.fuente AS source,
                   'texto' AS type, null AS image_path, 0.49 AS similarity
            LIMIT $top_k
        """
        params["top_k"] = top_k
        try:
            return await self.run(query, params)
        except Exception as e:
            print(f"⚠️ Entity search error: {e}")
            return []

    async def expand_neighborhood(self, node_ids: List[str],
                                 depth: int = NEO4J_GRAPH_DEPTH) -> List[Dict]:
        if not node_ids:
            return []
        query = """
            UNWIND $ids AS nid
            MATCH (n {id: nid})

            OPTIONAL MATCH (n)-[:PERTENECE_A]->(pdf:PDF)<-[:PERTENECE_A]-(vecino_pdf)
            WHERE vecino_pdf.id <> nid
            WITH n, nid, collect(DISTINCT vecino_pdf)[..5] AS list_pdf

            OPTIONAL MATCH (n)-[:MENCIONA]->(entidad)<-[:MENCIONA]-(vecino_entidad:Chunk)
            WHERE vecino_entidad.id <> nid
            WITH n, nid, list_pdf, collect(DISTINCT vecino_entidad)[..5] AS list_ent

            OPTIONAL MATCH (n)-[:SIMILAR_A]-(vecino_similar:Imagen)
            WITH n, nid, list_pdf, list_ent, collect(DISTINCT vecino_similar)[..5] AS list_sim

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
                END AS text,
                v.fuente AS source,
                CASE WHEN v:Imagen THEN 'imagen' ELSE 'texto' END AS type,
                CASE WHEN v:Imagen THEN v.path ELSE null END AS image_path,
                CASE 
                    WHEN (n:Imagen AND v:Imagen AND n.pagina = v.pagina) OR
                         (n:Chunk AND v:Imagen AND n.fuente = v.fuente)
                    THEN 0.95 
                    ELSE 0.3 
                END AS similarity
            LIMIT 15
        """
        try:
            return await self.run(query, {"ids": node_ids})
        except Exception as e:
            print(f"⚠️ Neighborhood expansion error: {e}")
            return []

    async def semantic_path_search(self,
                                         source_tissue: Optional[str],
                                         dest_tissue: Optional[str]) -> List[Dict]:
        if not source_tissue or not dest_tissue:
            return []
        query = """
            MATCH (origen {nombre: $source}), (destino {nombre: $dest})
            MATCH path = shortestPath((origen)-[*1..4]-(destino))
            UNWIND nodes(path) AS nodo
            OPTIONAL MATCH (c:Chunk)-[:MENCIONA]->(nodo)
            RETURN DISTINCT c.id AS id, c.texto AS text, c.fuente AS source,
                   'texto' AS type, null AS image_path, 0.4 AS similarity
            LIMIT 5
        """
        try:
            return await self.run(query, {"source": source_tissue, "dest": dest_tissue})
        except Exception as e:
            print(f"⚠️ Semantic Path search error: {e}")
            return []

    async def hybrid_search(self,
                                text_embedding: Optional[List[float]],
                                image_embedding_uni: Optional[List[float]],
                                image_embedding_plip: Optional[List[float]],
                                entities: Dict[str, List[str]],
                                top_k: int = 10) -> List[Dict]:
        res_text = []
        res_uni  = []
        res_plip = []
        res_ent  = []
        res_vec  = []

        if text_embedding is not None:
            res_text = await self.vector_search(text_embedding, TEXT_INDEX_NAME, top_k)
        if image_embedding_uni is not None:
            res_uni = await self.vector_search(image_embedding_uni, UNI_INDEX_NAME, top_k)
        if image_embedding_plip is not None:
            res_plip = await self.vector_search(image_embedding_plip, PLIP_INDEX_NAME, top_k)

        res_ent = await self.entity_search(entities, top_k)

        all_res = res_text + res_uni + res_plip
        top_ids = [r["id"] for r in all_res[:6] if r.get("id")]
        if top_ids:
            res_vec = await self.expand_neighborhood(top_ids)

        combined: Dict[str, Dict] = {}

        def add(results: List[Dict], weight: float):
            for r in results:
                key = r.get("id") or f"{r.get('source')}_{str(r.get('text',''))[:40]}"
                if not r.get("text") and not r.get("image_path"):
                    continue
                weighted_sim = r.get("similarity", 0) * weight
                if key not in combined:
                    combined[key] = {**r, "similarity": weighted_sim}
                else:
                    combined[key]["similarity"] += weighted_sim

        add(res_text, 0.80) 
        add(res_uni,  0.50) 
        add(res_plip, 0.50) 
        add(res_ent,  0.60) 
        add(res_vec,  0.20)

        final = sorted(combined.values(), key=lambda x: x["similarity"], reverse=True)

        print(f"   📊 Hybrid Search: Txt={len(res_text)} | "
              f"UNI={len(res_uni)} | PLIP={len(res_plip)} | Ent={len(res_ent)} | Vec={len(res_vec)} -> {len(final)}")

        return final[:15]
