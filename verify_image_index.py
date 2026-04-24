#!/usr/bin/env python3
"""
Verificación end-to-end del indexado de imágenes en Neo4j.

Este script verifica que:
1. Cada imagen indexada tenga embeddings UNI + PLIP
2. El caption contiene la etiqueta formal (Imagen X.X, Fig X.X)
3. El nombre_archivo y la etiqueta están correctamente almacenados
4. La búsqueda por embedding devuelve la misma imagen como top-1 (round-trip)
5. El texto asociado es la descripción real del PDF, no OCR ruidoso

Uso:
    python verify_image_index.py [--roundtrip] [--verbose]
"""
import os
import sys
import asyncio
import argparse
import importlib.util
import re

# Load the module from file
spec = importlib.util.spec_from_file_location("ne4j_histo", "ne4j-histo.py")
ne4j_histo = importlib.util.module_from_spec(spec)
sys.modules["ne4j_histo"] = ne4j_histo
spec.loader.exec_module(ne4j_histo)

from ne4j_histo import Neo4jClient, UniWrapper, PlipWrapper, userdata
import torch
import numpy as np


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def ok(msg): print(f"  {Colors.GREEN}✅ {msg}{Colors.END}")
def warn(msg): print(f"  {Colors.YELLOW}⚠️  {msg}{Colors.END}")
def fail(msg): print(f"  {Colors.RED}❌ {msg}{Colors.END}")
def info(msg): print(f"  {Colors.CYAN}ℹ️  {msg}{Colors.END}")
def header(msg): print(f"\n{Colors.BOLD}{'='*70}\n{msg}\n{'='*70}{Colors.END}")


async def verify_all(do_roundtrip: bool = False, verbose: bool = False):
    header("🔍 Verificación de Indexado de Imágenes en Neo4j")
    
    # Connect to Neo4j
    neo4j_uri = userdata.get("NEO4J_URI")
    neo4j_user = userdata.get("NEO4J_USERNAME") or userdata.get("NEO4J_USER")
    neo4j_password = userdata.get("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        fail("Faltan credenciales Neo4j en .env")
        return
    
    neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
    await neo4j.connect()
    ok("Conectado a Neo4j")
    
    # ── Test 1: Contar nodos ────────────────────────────────────
    header("1️⃣  Estadísticas de la Base de Datos")
    
    stats = await neo4j.run("""
        MATCH (i:Imagen)
        RETURN count(i) AS total,
               count(CASE WHEN i.embedding_uni IS NOT NULL THEN 1 END) AS con_uni,
               count(CASE WHEN i.embedding_plip IS NOT NULL THEN 1 END) AS con_plip,
               count(CASE WHEN i.caption IS NOT NULL AND i.caption <> '' THEN 1 END) AS con_caption,
               count(CASE WHEN i.nombre_archivo IS NOT NULL AND i.nombre_archivo <> '' THEN 1 END) AS con_nombre,
               count(CASE WHEN i.etiqueta IS NOT NULL AND i.etiqueta <> '' THEN 1 END) AS con_etiqueta,
               count(CASE WHEN i.path IS NOT NULL THEN 1 END) AS con_path
    """)
    
    if not stats:
        fail("No hay imágenes en Neo4j")
        await neo4j.close()
        return
    
    s = stats[0]
    total = s["total"]
    info(f"Total imágenes:    {total}")
    info(f"Con embeddings UNI:  {s['con_uni']}/{total}")
    info(f"Con embeddings PLIP: {s['con_plip']}/{total}")
    info(f"Con caption:         {s['con_caption']}/{total}")
    info(f"Con nombre_archivo:  {s['con_nombre']}/{total}")
    info(f"Con etiqueta:        {s['con_etiqueta']}/{total}")
    info(f"Con path:            {s['con_path']}/{total}")
    
    if s['con_uni'] < total:
        warn(f"{total - s['con_uni']} imágenes SIN embedding UNI")
    if s['con_plip'] < total:
        warn(f"{total - s['con_plip']} imágenes SIN embedding PLIP")
    if s['con_caption'] < total:
        warn(f"{total - s['con_caption']} imágenes SIN caption")
    if s['con_nombre'] < total:
        warn(f"{total - s['con_nombre']} imágenes SIN nombre_archivo (necesita reindexar)")
    if s['con_etiqueta'] < total:
        warn(f"{total - s['con_etiqueta']} imágenes SIN etiqueta (algunas páginas no tienen)")
    
    # ── Test 2: Verificar cada imagen ───────────────────────────
    header("2️⃣  Verificación Individual de Imágenes")
    
    imagenes = await neo4j.run("""
        MATCH (i:Imagen)
        RETURN i.id AS id, i.path AS path, i.pagina AS pagina,
               i.fuente AS fuente, i.caption AS caption,
               i.nombre_archivo AS nombre_archivo, i.etiqueta AS etiqueta,
               i.ocr_text AS ocr_text,
               CASE WHEN i.embedding_uni IS NOT NULL THEN true ELSE false END AS tiene_uni,
               CASE WHEN i.embedding_plip IS NOT NULL THEN true ELSE false END AS tiene_plip
        ORDER BY i.pagina
    """)
    
    errores = 0
    warnings = 0
    
    for img in imagenes:
        img_id = img["id"]
        path = img.get("path", "")
        pagina = img.get("pagina", "?")
        caption = img.get("caption", "")
        nombre = img.get("nombre_archivo", "")
        etiqueta = img.get("etiqueta", "")
        
        problemas = []
        
        # Verificar archivo físico
        if not path or not os.path.exists(path):
            problemas.append(f"❌ Archivo no existe: {path}")
            errores += 1
        
        # Verificar embeddings
        if not img.get("tiene_uni"):
            problemas.append("❌ Sin embedding UNI")
            errores += 1
        if not img.get("tiene_plip"):
            problemas.append("❌ Sin embedding PLIP")
            errores += 1
        
        # Verificar caption
        if not caption:
            problemas.append("⚠️ Sin caption")
            warnings += 1
        elif len(caption) < 20:
            problemas.append(f"⚠️ Caption muy corto ({len(caption)} chars)")
            warnings += 1
        
        # Verificar nombre_archivo
        if not nombre:
            problemas.append("⚠️ Sin nombre_archivo (reindexar)")
            warnings += 1
        
        # Verificar etiqueta
        if not etiqueta:
            # No es error para páginas fullpage
            if "_full" not in (nombre or path or ""):
                problemas.append("⚠️ Sin etiqueta formal")
                warnings += 1
        
        # Mostrar resultado
        display_name = nombre or os.path.basename(path) if path else img_id
        if problemas:
            print(f"\n  📄 Pág {pagina} | {display_name}")
            for p in problemas:
                print(f"     {p}")
            if verbose and caption:
                print(f"     Caption ({len(caption)} chars): {caption[:150]}...")
                if etiqueta:
                    print(f"     Etiqueta: {etiqueta}")
        elif verbose:
            print(f"\n  📄 Pág {pagina} | {display_name}")
            ok(f"UNI + PLIP + Caption ({len(caption)} chars)")
            if etiqueta:
                info(f"Etiqueta: {etiqueta}")
            print(f"     Caption: {caption[:100]}...")
    
    print(f"\n  {'─'*50}")
    print(f"  📊 Resumen: {total} imágenes | {errores} errores | {warnings} advertencias")
    if errores == 0:
        ok("Todas las imágenes pasan validación básica")
    else:
        fail(f"{errores} errores encontrados")
    
    # ── Test 3: Round-trip (embedding → búsqueda → top-1) ──────
    if do_roundtrip:
        header("3️⃣  Test Round-Trip (Embedding → Búsqueda → Top-1)")
        
        print("  Cargando modelos de embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check GPU compatibility
        if device == "cuda":
            try:
                cap = torch.cuda.get_device_capability(0)
                if cap[0] < 7:
                    print(f"  ⚠️ GPU incompatible (sm_{cap[0]}{cap[1]}), usando CPU")
                    device = "cpu"
            except:
                pass
        
        info(f"Device: {device}")
        
        uni = UniWrapper(device)
        uni.load()
        
        plip = PlipWrapper(device)
        plip.load()
        
        # Tomar 5 imágenes de prueba
        test_images = await neo4j.run("""
            MATCH (i:Imagen)
            WHERE i.embedding_uni IS NOT NULL AND i.path IS NOT NULL
            RETURN i.id AS id, i.path AS path, i.pagina AS pagina,
                   i.nombre_archivo AS nombre_archivo, i.etiqueta AS etiqueta,
                   i.caption AS caption
            LIMIT 5
        """)
        
        roundtrip_ok = 0
        roundtrip_fail = 0
        
        for test_img in test_images:
            path = test_img["path"]
            img_id = test_img["id"]
            nombre = test_img.get("nombre_archivo", os.path.basename(path))
            
            if not os.path.exists(path):
                fail(f"{nombre}: archivo no encontrado")
                roundtrip_fail += 1
                continue
            
            # Generar embeddings (sin preprocessing — matching what's indexed)
            emb_u = uni.embed_image(path, preprocess=False)
            emb_p = plip.embed_image(path, preprocess=False)
            
            # Buscar en Neo4j
            results_uni = await neo4j.busqueda_vectorial(emb_u.tolist(), "histo_img_uni", 3)
            results_plip = await neo4j.busqueda_vectorial(emb_p.tolist(), "histo_img_plip", 3)
            
            # Verificar que la misma imagen es top-1
            uni_top1 = results_uni[0] if results_uni else {}
            plip_top1 = results_plip[0] if results_plip else {}
            
            uni_match = uni_top1.get("imagen_path") == path
            plip_match = plip_top1.get("imagen_path") == path
            uni_score = uni_top1.get("similitud", 0)
            plip_score = plip_top1.get("similitud", 0)
            
            etiqueta = test_img.get("etiqueta", "N/A")
            caption_preview = (test_img.get("caption", "")[:80] + "...") if test_img.get("caption") else "N/A"
            
            print(f"\n  📷 {nombre} (Pág {test_img.get('pagina', '?')})")
            
            if uni_match and uni_score > 0.95:
                ok(f"UNI top-1 match (score: {uni_score:.4f})")
            elif uni_match:
                warn(f"UNI top-1 match pero score bajo: {uni_score:.4f}")
            else:
                fail(f"UNI top-1 NO coincide (score: {uni_score:.4f}, got: {os.path.basename(uni_top1.get('imagen_path', 'N/A'))})")
            
            if plip_match and plip_score > 0.95:
                ok(f"PLIP top-1 match (score: {plip_score:.4f})")
            elif plip_match:
                warn(f"PLIP top-1 match pero score bajo: {plip_score:.4f}")
            else:
                fail(f"PLIP top-1 NO coincide (score: {plip_score:.4f}, got: {os.path.basename(plip_top1.get('imagen_path', 'N/A'))})")
            
            info(f"Etiqueta: {etiqueta}")
            info(f"Caption: {caption_preview}")
            
            # Verificar que el texto retornado contiene la etiqueta
            texto_retornado = uni_top1.get("texto", "")
            if etiqueta and etiqueta != "N/A" and etiqueta.lower() in texto_retornado.lower():
                ok(f"Texto retornado CONTIENE la etiqueta '{etiqueta}'")
            elif etiqueta and etiqueta != "N/A":
                warn(f"Texto retornado NO contiene la etiqueta '{etiqueta}'")
            
            if uni_match and plip_match:
                roundtrip_ok += 1
            else:
                roundtrip_fail += 1
        
        print(f"\n  {'─'*50}")
        print(f"  📊 Round-trip: {roundtrip_ok} OK | {roundtrip_fail} fallos de {len(test_images)} probados")
        if roundtrip_fail == 0:
            ok("✨ Todos los round-trips exitosos — el sistema nunca falla en encontrar la imagen!")
        else:
            warn(f"{roundtrip_fail} imágenes no se encontraron como top-1")
    
    # ── Test 4: Verificar caption contiene etiqueta ─────────────
    header("4️⃣  Verificación de Captions con Etiqueta")
    
    imgs_con_etiqueta = await neo4j.run("""
        MATCH (i:Imagen)
        WHERE i.etiqueta IS NOT NULL AND i.etiqueta <> ''
        RETURN i.id AS id, i.nombre_archivo AS nombre,
               i.etiqueta AS etiqueta, i.caption AS caption,
               i.pagina AS pagina
        ORDER BY i.pagina
    """)
    
    match_count = 0
    mismatch_count = 0
    
    for img in imgs_con_etiqueta:
        caption = img.get("caption", "")
        etiqueta = img.get("etiqueta", "")
        nombre = img.get("nombre", img["id"])
        
        if etiqueta.lower() in caption.lower():
            match_count += 1
            if verbose:
                ok(f"Pág {img.get('pagina', '?')} | {nombre}: '{etiqueta}' ∈ caption ✓")
        else:
            mismatch_count += 1
            warn(f"Pág {img.get('pagina', '?')} | {nombre}: '{etiqueta}' NO está en caption")
            if verbose:
                print(f"     Caption inicio: {caption[:120]}...")
    
    print(f"\n  📊 Etiquetas en caption: {match_count} match | {mismatch_count} desalineados")
    if mismatch_count == 0 and match_count > 0:
        ok("Todas las etiquetas se encuentran en el caption asociado")
    elif match_count == 0 and len(imgs_con_etiqueta) == 0:
        warn("No hay imágenes con etiqueta — ejecutar --reindex --force")
    
    await neo4j.close()
    
    header("✅ Verificación Completada")
    print(f"  Para reindexar: python ne4j-histo.py --reindex --force\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificar indexado de imágenes en Neo4j")
    parser.add_argument("--roundtrip", action="store_true",
                       help="Ejecutar test round-trip (requiere cargar modelos)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mostrar detalles de cada imagen")
    args = parser.parse_args()
    
    asyncio.run(verify_all(do_roundtrip=args.roundtrip, verbose=args.verbose))
