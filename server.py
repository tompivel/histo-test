"""
Servidor FastAPI para RAG Histología Neo4j — Fullstack A2UI
============================================================
Wrappea AsistenteHistologiaNeo4j y expone endpoints REST + A2UI.
El módulo ne4j-histo.py se importa sin modificación.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

# ── Importar el módulo principal ─────────────────────────────────────
# ne4j-histo.py tiene guión, así que lo importamos con importlib
import importlib.util

_HISTO_PATH = Path(__file__).parent / "ne4j-histo.py"
spec = importlib.util.spec_from_file_location("ne4j_histo", str(_HISTO_PATH))
ne4j_histo = importlib.util.module_from_spec(spec)

# Prevenir que el módulo ejecute su __main__
_original_argv = sys.argv
sys.argv = ["ne4j-histo.py"]
spec.loader.exec_module(ne4j_histo)
sys.argv = _original_argv

AsistenteHistologiaNeo4j = ne4j_histo.AsistenteHistologiaNeo4j
DIRECTORIO_PDFS = ne4j_histo.DIRECTORIO_PDFS

# ── Estado global ────────────────────────────────────────────────────
asistente: Optional[AsistenteHistologiaNeo4j] = None
_init_complete = False
_init_error: Optional[str] = None


# ── Modelos Pydantic ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None


class ChatResponse(BaseModel):
    respuesta: str
    estructura_identificada: Optional[str] = None
    imagenes_recuperadas: list = []
    imagenes_base64: list = []  # Lista de {filename, base64, mime_type}
    trayectoria: list = []
    imagen_activa: Optional[str] = None
    mostrar_imagenes: bool = False


# ── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asistente, _init_complete, _init_error
    print("🚀 Iniciando servidor RAG Histología Neo4j + A2UI...")

    try:
        asistente = AsistenteHistologiaNeo4j()
        await asistente.inicializar_componentes()

        print("📚 Leyendo PDFs...")
        asistente.procesar_contenido_base(DIRECTORIO_PDFS)

        print("📋 Extrayendo temario...")
        await asistente.extraer_y_preparar_temario()
        n_temas = len(asistente.extractor_temario.temas) if asistente.extractor_temario else 0
        print(f"   → {n_temas} temas")

        print("💾 Verificando e indexando base de datos Neo4j (si está vacía)...")
        await asistente.indexar_en_neo4j(DIRECTORIO_PDFS, forzar=False)

        _init_complete = True
        print("✅ Servidor listo")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _init_error = str(e)
        print(f"❌ Error inicializando: {e}")

    yield

    # Shutdown
    if asistente:
        await asistente.cerrar()
    print("👋 Servidor apagado")


# ── App FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Histología Neo4j + A2UI",
    description="Sistema RAG Multimodal de Histología — Fullstack",
    version="4.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Archivos estáticos del cliente
CLIENT_DIR = Path(__file__).parent / "client"

# Directorio de imágenes extraídas (para servir al frontend)
IMAGENES_DIR = Path(__file__).parent / "imagenes_extraidas"


def _check_ready():
    if not _init_complete:
        raise HTTPException(503, detail=_init_error or "Sistema inicializándose...")


# ── Rutas de archivos del frontend ───────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(CLIENT_DIR / "index.html"))


@app.get("/app.js")
async def serve_js():
    return FileResponse(str(CLIENT_DIR / "app.js"), media_type="application/javascript")


@app.get("/style.css")
async def serve_css():
    return FileResponse(str(CLIENT_DIR / "style.css"), media_type="text/css")


# ── API: Estado ──────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    if not _init_complete:
        return {
            "ready": False,
            "error": _init_error,
        }
    return {
        "ready": True,
        "n_temas": len(asistente.extractor_temario.temas) if asistente.extractor_temario else 0,
        "imagen_activa": os.path.basename(asistente.memoria.get_imagen_activa())
            if asistente.memoria and asistente.memoria.get_imagen_activa() else None,
        "turno": asistente.memoria.turno_actual if asistente.memoria else 0,
        "device": asistente.device,
    }


# ── API: Temario ─────────────────────────────────────────────────────
@app.get("/api/temario")
async def get_temario():
    _check_ready()
    temas = asistente.extractor_temario.temas if asistente.extractor_temario else []
    return {"temas": temas, "total": len(temas)}


# ── API: Chat (texto plano) ─────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest):
    _check_ready()

    imagen_path = None
    try:
        # Si hay imagen, guardarla en un directorio persistente
        if req.image_base64:
            chat_img_dir = Path(__file__).parent / "imagenes_chat"
            chat_img_dir.mkdir(exist_ok=True)
            
            ext = ".png"
            if req.image_filename:
                _, ext = os.path.splitext(req.image_filename)
                if not ext:
                    ext = ".png"
            
            # Nombre de archivo único
            import uuid
            nombre_archivo = f"upload_{uuid.uuid4().hex[:8]}{ext}"
            imagen_path = str(chat_img_dir / nombre_archivo)
            
            with open(imagen_path, "wb") as f:
                f.write(base64.b64decode(req.image_base64))
                
            print(f"📷 Imagen guardada para chat: {imagen_path}")

        # Ejecutar consulta RAG
        respuesta = await asistente.consultar(
            consulta_texto=req.query,
            imagen_path=imagen_path,
        )

        # Leer resultado directo del asistente (más confiable que el archivo JSON)
        resultado_directo = getattr(asistente, '_ultimo_resultado', {})
        mostrar_imgs = resultado_directo.get("mostrar_imagenes", False)
        imagenes_rec_directas = resultado_directo.get("imagenes_recuperadas", [])
        estructura = resultado_directo.get("estructura_identificada")

        # Convertir imágenes recuperadas a base64 para el frontend
        imagenes_b64 = []
        for img_path in imagenes_rec_directas[:1]:  # Solo la de mayor similitud
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        data_b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(img_path)[1].lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    imagenes_b64.append({
                        "filename": os.path.basename(img_path),
                        "base64": data_b64,
                        "mime_type": mime
                    })
                except Exception as e:
                    print(f"⚠️ Error leyendo imagen {img_path}: {e}")

        # Solo buscar imagen en Neo4j si el usuario la pidió explícitamente
        if not imagenes_b64 and mostrar_imgs and hasattr(asistente, 'neo4j') and asistente.neo4j:
            import re
            import unicodedata
            
            def sin_tildes(s):
                return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            
            # Extraer términos de búsqueda de la consulta original
            palabras_ignorar = {"imagen", "imagenes", "foto", "mostrame", "mostra", "quiero",
                               "dame", "pasame", "una", "ver", "del", "las", "los", "que", "como"}
            terminos = [w.strip().lower() for w in req.query.split() 
                       if len(w.strip()) > 3 and w.strip().lower() not in palabras_ignorar]
            # Agregar variantes sin tildes
            terminos_expandidos = list(set(terminos + [sin_tildes(t) for t in terminos]))
            
            if terminos_expandidos:
                try:
                    # Búsqueda precisa: rankear imágenes por cantidad de términos que matchean en caption
                    neo_res = await asistente.neo4j.run(
                        """MATCH (i:Imagen) WHERE i.path IS NOT NULL
                        WITH i, toLower(coalesce(i.caption,'')) AS cap
                        WITH i, cap, 
                             size([t IN $terminos WHERE cap CONTAINS t]) AS matches
                        WHERE matches > 0
                        RETURN i.path AS path, coalesce(i.caption,'') AS caption, matches
                        ORDER BY matches DESC
                        LIMIT 1""",
                        {"terminos": terminos_expandidos}
                    )
                    # Fallback: buscar en ocr_text si no hay resultado por caption
                    if not neo_res:
                        neo_res = await asistente.neo4j.run(
                            """MATCH (i:Imagen) WHERE i.path IS NOT NULL
                            WITH i, toLower(coalesce(i.ocr_text,'')) AS ocr
                            WITH i, ocr,
                                 size([t IN $terminos WHERE ocr CONTAINS t]) AS matches
                            WHERE matches > 0
                            RETURN i.path AS path, coalesce(i.caption,'') AS caption, matches
                            ORDER BY matches DESC
                            LIMIT 1""",
                            {"terminos": terminos_expandidos}
                        )
                    for r in neo_res:
                        p = r.get("path")
                        if p and os.path.exists(p):
                            with open(p, "rb") as f:
                                data_b64 = base64.b64encode(f.read()).decode("utf-8")
                            ext = os.path.splitext(p)[1].lower()
                            mime = "image/png" if ext == ".png" else "image/jpeg"
                            imagenes_b64.append({
                                "filename": os.path.basename(p),
                                "base64": data_b64,
                                "mime_type": mime
                            })
                            print(f"   🖼️ Imagen encontrada por entidad: {os.path.basename(p)}")
                except Exception as e:
                    print(f"⚠️ Error buscando imágenes por entidades: {e}")
            
            # Fallback: buscar por regex "Imagen X.Y" en la respuesta del LLM
            if not imagenes_b64 and respuesta:
                refs = re.findall(r'[Ii]magen\s+(\d+[\.\s]*\d*)', respuesta)
                archivos_vistos = set()
                for ref_raw in refs[:1]:  # Solo la primera referencia
                    ref_clean = ref_raw.strip().replace(" ", ".")
                    caption_q = f"imagen {ref_clean}".lower()
                    try:
                        neo_res = await asistente.neo4j.run(
                            "MATCH (i:Imagen) WHERE toLower(coalesce(i.caption,'')) CONTAINS $q "
                            "RETURN i.path AS path LIMIT 1",
                            {"q": caption_q}
                        )
                        if neo_res:
                            p = neo_res[0].get("path")
                            if p and os.path.exists(p) and p not in archivos_vistos:
                                archivos_vistos.add(p)
                                with open(p, "rb") as f:
                                    data_b64 = base64.b64encode(f.read()).decode("utf-8")
                                ext = os.path.splitext(p)[1].lower()
                                mime = "image/png" if ext == ".png" else "image/jpeg"
                                imagenes_b64.append({
                                    "filename": os.path.basename(p),
                                    "base64": data_b64,
                                    "mime_type": mime
                                })
                                print(f"   🖼️ Imagen por referencia: Imagen {ref_clean} → {os.path.basename(p)}")
                    except Exception as e:
                        print(f"⚠️ Error buscando ref '{ref_raw}': {e}")

        # Leer trayectoria del archivo (solo para metadata de debug)
        trayectoria = []
        trayectoria_file = Path(__file__).parent / "trayectoria_neo4j.json"
        if trayectoria_file.exists():
            try:
                with open(trayectoria_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trayectoria = data.get("trayectoria", [])
            except Exception:
                pass

        img_activa = None
        if asistente.memoria and asistente.memoria.get_imagen_activa():
            img_activa = os.path.basename(asistente.memoria.get_imagen_activa())

        return ChatResponse(
            respuesta=respuesta,
            estructura_identificada=estructura,
            imagenes_recuperadas=[os.path.basename(p) for p in imagenes_rec_directas],
            imagenes_base64=imagenes_b64,
            trayectoria=trayectoria,
            imagen_activa=img_activa,
            mostrar_imagenes=mostrar_imgs,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))
    finally:
        pass



# ── API: Limpiar imagen ──────────────────────────────────────────────
@app.post("/api/imagen/limpiar")
async def limpiar_imagen():
    _check_ready()
    if asistente.memoria:
        asistente.memoria.set_imagen(None)
    return {"ok": True, "mensaje": "Imagen activa eliminada"}


# ── Ruta estática: imágenes extraídas ────────────────────────────────
if IMAGENES_DIR.exists():
    app.mount("/imagenes_extraidas", StaticFiles(directory=str(IMAGENES_DIR)), name="imagenes_extraidas")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    port = int(os.getenv("PORT", "10005"))
    print(f"🌐 Servidor en http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
