"""Debug: Compare what's stored in Neo4j for arch2_pag5 and arch2_pag12"""
import asyncio
import os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv()

async def main():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
    
    async with driver.session() as session:
        # Get info for pag5 and pag12
        for pag_name in ["arch2_pag5", "arch2_pag12"]:
            print(f"\n{'='*60}")
            print(f"  {pag_name}")
            print(f"{'='*60}")
            
            result = await session.run(
                """
                MATCH (i:Imagen) 
                WHERE i.nombre_archivo CONTAINS $name
                RETURN i.nombre_archivo AS nombre,
                       i.pagina AS pagina,
                       i.fuente AS fuente,
                       i.etiqueta AS etiqueta,
                       substring(coalesce(i.caption, ''), 0, 400) AS caption,
                       substring(coalesce(i.texto_pagina, ''), 0, 400) AS texto_pagina,
                       substring(coalesce(i.ocr_text, ''), 0, 200) AS ocr_text,
                       i.path AS path
                """,
                {"name": pag_name}
            )
            records = await result.data()
            
            if not records:
                print(f"  ⚠️ NOT FOUND in Neo4j!")
            else:
                for r in records:
                    print(f"  nombre_archivo: {r['nombre']}")
                    print(f"  pagina: {r['pagina']}")
                    print(f"  fuente: {r['fuente']}")
                    print(f"  etiqueta: {r['etiqueta']}")
                    print(f"  path: {r['path']}")
                    print(f"\n  --- CAPTION (first 400 chars) ---")
                    print(f"  {r['caption']}")
                    print(f"\n  --- TEXTO_PAGINA (first 400 chars) ---")
                    print(f"  {r['texto_pagina']}")
                    print(f"\n  --- OCR (first 200 chars) ---")
                    print(f"  {r['ocr_text']}")
        
        # Also check: what text is returned by vector search for these images
        print(f"\n{'='*60}")
        print("  Checking hybrid search text for image nodes...")
        print(f"{'='*60}")
        
        result = await session.run("""
            MATCH (i:Imagen) 
            WHERE i.nombre_archivo CONTAINS 'arch2_pag5' OR i.nombre_archivo CONTAINS 'arch2_pag12'
            RETURN i.nombre_archivo AS nombre,
                   CASE 
                       WHEN i.caption IS NOT NULL AND i.caption <> '' THEN 'CAPTION'
                       WHEN i.texto_pagina IS NOT NULL AND i.texto_pagina <> '' THEN 'TEXTO_PAGINA'
                       WHEN i.ocr_text IS NOT NULL AND i.ocr_text <> '' THEN 'OCR_TEXT'
                       ELSE 'EMPTY'
                   END AS text_source,
                   CASE 
                       WHEN i.caption IS NOT NULL AND i.caption <> '' THEN substring(i.caption, 0, 300)
                       WHEN i.texto_pagina IS NOT NULL AND i.texto_pagina <> '' THEN substring(i.texto_pagina, 0, 300)
                       WHEN i.ocr_text IS NOT NULL AND i.ocr_text <> '' THEN substring(i.ocr_text, 0, 300)
                       ELSE ''
                   END AS texto_usado
        """)
        records = await result.data()
        for r in records:
            print(f"\n  {r['nombre']} → uses {r['text_source']}:")
            print(f"  {r['texto_usado']}")
    
    await driver.close()

asyncio.run(main())
