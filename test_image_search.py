#!/usr/bin/env python3
"""
Diagnostic script to test image embedding search in Neo4j
"""
import os
import sys
import asyncio
import importlib.util
from pathlib import Path

# Load the module from file
spec = importlib.util.spec_from_file_location("ne4j_histo", "ne4j-histo.py")
ne4j_histo = importlib.util.module_from_spec(spec)
sys.modules["ne4j_histo"] = ne4j_histo
spec.loader.exec_module(ne4j_histo)

from ne4j_histo import Neo4jClient, UniWrapper, PlipWrapper, userdata
import torch
import numpy as np

async def test_image_search():
    """Test image embedding search"""
    print("="*80)
    print("🔍 Testing Image Embedding Search in Neo4j")
    print("="*80)
    
    # Initialize Neo4j client
    neo4j_uri = userdata.get("NEO4J_URI")
    neo4j_user = userdata.get("NEO4J_USERNAME") or userdata.get("NEO4J_USER")
    neo4j_password = userdata.get("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("❌ Missing Neo4j credentials in .env")
        return
    
    print(f"\n1️⃣ Connecting to Neo4j: {neo4j_uri}")
    neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    await neo4j.connect()
    print("   ✅ Connected")
    
    # Check if images exist in Neo4j
    print("\n2️⃣ Checking images in Neo4j...")
    query = """
    MATCH (i:Imagen)
    RETURN count(i) as total,
           count(i.embedding_uni) as with_uni,
           count(i.embedding_plip) as with_plip,
           i.fuente as pdf_name
    LIMIT 1
    """
    result = await neo4j.run(query)
    if result:
        print(f"   📊 Total images: {result[0].get('total', 0)}")
        print(f"   📊 With UNI embeddings: {result[0].get('with_uni', 0)}")
        print(f"   📊 With PLIP embeddings: {result[0].get('with_plip', 0)}")
        print(f"   📄 PDF: {result[0].get('pdf_name', 'N/A')}")
    else:
        print("   ⚠️ No images found in Neo4j")
        await neo4j.close()
        return
    
    # Get a sample image from Neo4j
    print("\n3️⃣ Getting sample image from Neo4j...")
    query = """
    MATCH (i:Imagen)
    WHERE i.embedding_uni IS NOT NULL AND i.path IS NOT NULL
    RETURN i.path as path, i.pagina as page, i.caption as caption
    LIMIT 1
    """
    result = await neo4j.run(query)
    if not result or not result[0].get('path'):
        print("   ⚠️ No images with embeddings found")
        await neo4j.close()
        return
    
    sample_image_path = result[0]['path']
    sample_page = result[0]['page']
    sample_caption = result[0].get('caption', '')[:100]
    
    print(f"   📷 Sample image: {sample_image_path}")
    print(f"   📄 Page: {sample_page}")
    print(f"   📝 Caption: {sample_caption}...")
    
    if not os.path.exists(sample_image_path):
        print(f"   ❌ Image file not found: {sample_image_path}")
        await neo4j.close()
        return
    
    # Initialize embedding models
    print("\n4️⃣ Initializing embedding models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   🖥️ Device: {device}")
    
    uni = UniWrapper(device)
    uni.load()
    
    plip = PlipWrapper(device)
    plip.load()
    
    # Generate embeddings for the sample image
    print("\n5️⃣ Generating embeddings for sample image...")
    emb_uni = uni.embed_image(sample_image_path)
    emb_plip = plip.embed_image(sample_image_path)
    
    print(f"   📊 UNI embedding shape: {emb_uni.shape}")
    print(f"   📊 PLIP embedding shape: {emb_plip.shape}")
    print(f"   📊 UNI embedding norm: {np.linalg.norm(emb_uni):.4f}")
    print(f"   📊 PLIP embedding norm: {np.linalg.norm(emb_plip):.4f}")
    
    # Search using UNI embeddings
    print("\n6️⃣ Searching with UNI embeddings...")
    results_uni = await neo4j.busqueda_vectorial(
        emb_uni.tolist(), 
        "histo_img_uni", 
        top_k=5
    )
    
    print(f"   📊 Found {len(results_uni)} results")
    for i, r in enumerate(results_uni[:3], 1):
        img_path = r.get('imagen_path', 'N/A')
        score = r.get('similitud', 0)
        page = r.get('fuente', 'N/A')
        print(f"   {i}. Score: {score:.4f} | Page: {page} | Path: {os.path.basename(img_path)}")
        
        # Check if it's the same image
        if img_path == sample_image_path:
            print(f"      ✅ MATCH! Found the same image (score: {score:.4f})")
    
    # Search using PLIP embeddings
    print("\n7️⃣ Searching with PLIP embeddings...")
    results_plip = await neo4j.busqueda_vectorial(
        emb_plip.tolist(), 
        "histo_img_plip", 
        top_k=5
    )
    
    print(f"   📊 Found {len(results_plip)} results")
    for i, r in enumerate(results_plip[:3], 1):
        img_path = r.get('imagen_path', 'N/A')
        score = r.get('similitud', 0)
        page = r.get('fuente', 'N/A')
        print(f"   {i}. Score: {score:.4f} | Page: {page} | Path: {os.path.basename(img_path)}")
        
        # Check if it's the same image
        if img_path == sample_image_path:
            print(f"      ✅ MATCH! Found the same image (score: {score:.4f})")
    
    # Test hybrid search
    print("\n8️⃣ Testing hybrid search...")
    results_hybrid = await neo4j.busqueda_hibrida(
        texto_embedding=None,
        imagen_embedding_uni=emb_uni.tolist(),
        imagen_embedding_plip=emb_plip.tolist(),
        entidades={},
        top_k=5
    )
    
    print(f"   📊 Found {len(results_hybrid)} results")
    for i, r in enumerate(results_hybrid[:3], 1):
        img_path = r.get('imagen_path', 'N/A')
        score = r.get('similitud', 0)
        tipo = r.get('tipo', 'N/A')
        fuente = r.get('fuente', 'N/A')
        print(f"   {i}. Score: {score:.4f} | Type: {tipo} | Source: {fuente}")
        if img_path and img_path != 'N/A':
            print(f"      Path: {os.path.basename(img_path)}")
            if img_path == sample_image_path:
                print(f"      ✅ MATCH! Found the same image (hybrid score: {score:.4f})")
    
    # Test with a different image
    print("\n9️⃣ Testing with a different image...")
    query = """
    MATCH (i:Imagen)
    WHERE i.embedding_uni IS NOT NULL AND i.path IS NOT NULL AND i.path <> $exclude_path
    RETURN i.path as path, i.pagina as page
    LIMIT 1
    """
    result = await neo4j.run(query, {"exclude_path": sample_image_path})
    
    if result and result[0].get('path'):
        different_image_path = result[0]['path']
        different_page = result[0]['page']
        
        print(f"   📷 Different image: {different_image_path}")
        print(f"   📄 Page: {different_page}")
        
        if os.path.exists(different_image_path):
            # Generate embeddings for different image
            emb_uni_diff = uni.embed_image(different_image_path)
            emb_plip_diff = plip.embed_image(different_image_path)
            
            # Search with different image
            results_diff = await neo4j.busqueda_hibrida(
                texto_embedding=None,
                imagen_embedding_uni=emb_uni_diff.tolist(),
                imagen_embedding_plip=emb_plip_diff.tolist(),
                entidades={},
                top_k=5
            )
            
            print(f"   📊 Found {len(results_diff)} results")
            for i, r in enumerate(results_diff[:3], 1):
                img_path = r.get('imagen_path', 'N/A')
                score = r.get('similitud', 0)
                tipo = r.get('tipo', 'N/A')
                print(f"   {i}. Score: {score:.4f} | Type: {tipo}")
                if img_path and img_path != 'N/A':
                    print(f"      Path: {os.path.basename(img_path)}")
                    if img_path == different_image_path:
                        print(f"      ✅ MATCH! Found the same image (score: {score:.4f})")
    
    await neo4j.close()
    
    print("\n" + "="*80)
    print("✅ Diagnostic complete!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_image_search())
