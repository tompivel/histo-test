"""
FastAPI Server for Histology Neo4j RAG — Fullstack A2UI (v4.4)
============================================================
Instantiates the `Neo4jHistologyAgent` from the new modular architecture
and exposes the chat interface (frontend) and async APIs for the RAG.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import uuid
from PIL import Image
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
import uvicorn

from dotenv import load_dotenv

# ── Imports from the newly refactored English architecture v4.4 ───────
from core.agent import Neo4jHistologyAgent
from utils.config import userdata

load_dotenv()

# ── Global State ─────────────────────────────────────────────────────
agent: Optional[Neo4jHistologyAgent] = None
_init_complete = False
_init_error: Optional[str] = None

# Visual session state proxy for FastAPI (since Core Agent v4.4 is stateless)
active_image_state_path: Optional[str] = None
active_image_state_name: Optional[str] = None


# ── Pydantic Models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    session_id: str = "web_session_alpha"
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None

class ChatResponse(BaseModel):
    respuesta: str
    estructura_identificada: Optional[str] = None
    imagenes_recuperadas: list = []
    trayectoria: list = []
    imagen_activa: Optional[str] = None


# ── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, _init_complete, _init_error
    print("🚀 Starting Histology Neo4j RAG Server Modular (v4.4)...")

    try:
        neo4j_uri = userdata.get('NEO4J_URI') or "bolt://localhost:7687"
        neo4j_user = userdata.get('NEO4J_USERNAME') or "neo4j"
        neo4j_pass = userdata.get('NEO4J_PASSWORD') or "password"
        
        agent = Neo4jHistologyAgent(neo4j_uri, neo4j_user, neo4j_pass)
        print("⚙️ Injecting Models and verifying Neo4j Constraints...")
        await agent.initialize()

        _init_complete = True
        print("✅ Uvicorn Server Ready.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _init_error = str(e)
        print(f"❌ Error initializing: {e}")

    yield

    if agent:
        await agent.db.close()
    print("👋 Server Offline")


# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Histología Neo4j + Frontend GUI",
    description="Sistema RAG Multimodal de Histología — Modular v4.4",
    version="4.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT_DIR = Path(__file__).parent / "client"

def _check_ready():
    if not _init_complete:
        raise HTTPException(503, detail=_init_error or "System Initializing... please wait")

# ── Frontend File Routes ─────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(CLIENT_DIR / "index.html"))

@app.get("/app.js")
async def serve_js():
    return FileResponse(str(CLIENT_DIR / "app.js"), media_type="application/javascript")

@app.get("/style.css")
async def serve_css():
    return FileResponse(str(CLIENT_DIR / "style.css"), media_type="text/css")


# ── API: Status ──────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    if not _init_complete:
        return {"ready": False, "error": _init_error}
        
    topics = agent.classifier.syllabus if (agent and agent.classifier) else []
    return {
        "ready": True,
        "n_temas": len(topics),
        "imagen_activa": active_image_state_name,
        "turno": agent.memory.msg_count if (agent and agent.memory) else 0,
        "device": agent.device if agent else "cpu",
    }


# ── API: Temario (Syllabus) ──────────────────────────────────────────
@app.get("/api/temario")
async def get_temario():
    _check_ready()
    topics = agent.classifier.syllabus if (agent and agent.classifier) else []
    return {"temas": topics, "total": len(topics)}


# ── API: Chat (RAG Engine) ───────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest):
    global active_image_state_path, active_image_state_name
    _check_ready()

    try:
        if req.image_base64:
            chat_img_dir = Path(__file__).parent / "imagenes_chat"
            chat_img_dir.mkdir(exist_ok=True)
            
            ext = ".png"
            if req.image_filename:
                _, ext = os.path.splitext(req.image_filename)
                if not ext: ext = ".png"
            
            filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
            img_path = str(chat_img_dir / filename)
            
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(req.image_base64))
                
            active_image_state_path = img_path
            active_image_state_name = req.image_filename or filename
            print(f"📷 Image loaded to backend state: {active_image_state_name}")

        active_img_pil = None
        if active_image_state_path and os.path.exists(active_image_state_path):
            active_img_pil = Image.open(active_image_state_path).convert('RGB')

        agent_result = await agent.query(
            text_query=req.query,
            session_id=req.session_id,
            image=active_img_pil,
            history=[]
        )

        return ChatResponse(
            respuesta=agent_result.get("answer", ""),
            estructura_identificada=agent_result.get("identified_structure"),
            imagenes_recuperadas=agent_result.get("recovered_images", []),
            trayectoria=agent_result.get("trajectory", []),
            imagen_activa=active_image_state_name,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


# ── API: Clear Image ─────────────────────────────────────────────────
@app.post("/api/imagen/limpiar")
async def limpiar_imagen():
    global active_image_state_path, active_image_state_name
    _check_ready()
    active_image_state_path = None
    active_image_state_name = None
    return {"ok": True, "mensaje": "Active image decoupled"}


def main():
    port = int(os.getenv("PORT", "10005"))
    print(f"🌐 Fullstack Server listening at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
