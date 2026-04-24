"""Quick check if Neo4j has data now"""
import asyncio, os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
load_dotenv()

async def main():
    driver = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    async with driver.session() as s:
        # Count nodes
        r1 = await s.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
        data = [dict(r) async for r in r1]
        if data:
            print("NODOS EN NEO4J:")
            for d in data:
                print(f"  {d['label']}: {d['cnt']}")
        else:
            print("❌ BASE DE DATOS VACÍA")
            
        # Check for arch2_pag1 specifically
        r2 = await s.run("MATCH (i:Imagen) WHERE i.nombre_archivo CONTAINS 'arch2_pag1' RETURN i.nombre_archivo AS name, i.etiqueta AS etiq, substring(coalesce(i.caption,''), 0, 200) AS cap")
        imgs = [dict(r) async for r in r2]
        if imgs:
            print(f"\narch2_pag1 encontrado:")
            for i in imgs:
                print(f"  Nombre: {i['name']}")
                print(f"  Etiqueta: {i['etiq']}")
                print(f"  Caption: {i['cap']}")
        else:
            print("\n⚠️ arch2_pag1 NO encontrado en Neo4j")
    await driver.close()

asyncio.run(main())
