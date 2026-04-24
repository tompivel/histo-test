import sys
import asyncio
import os
import importlib.util

spec = importlib.util.spec_from_file_location("ne4j_histo", "ne4j-histo.py")
ne4j_histo = importlib.util.module_from_spec(spec)
sys.modules["ne4j_histo"] = ne4j_histo
spec.loader.exec_module(ne4j_histo)

from ne4j_histo import Neo4jClient, userdata

async def run():
    neo4j = Neo4jClient(userdata.get("NEO4J_URI"), userdata.get("NEO4J_USERNAME"), userdata.get("NEO4J_PASSWORD"))
    await neo4j.connect()
    
    # Let's get the embeddings for arch2_pag1.png
    query = """
    MATCH (i:Imagen)
    WHERE i.path CONTAINS "arch2_pag1.png"
    RETURN i.embedding_uni AS emb_uni, i.embedding_plip AS emb_plip
    """
    res = await neo4j.run(query)
    if not res:
        print("arch2_pag1.png not found")
        await neo4j.close()
        return
        
    emb_uni = res[0]["emb_uni"]
    
    print("Searching UNI for arch2_pag1.png")
    results = await neo4j.busqueda_vectorial(emb_uni, "histo_img_uni", 10)
    for i, r in enumerate(results):
        print(f"{i+1}. Score: {r['similitud']:.4f} | Path: {os.path.basename(r['imagen_path'])} | Caption: {r['texto'][:100]}")
        
    await neo4j.close()

asyncio.run(run())
