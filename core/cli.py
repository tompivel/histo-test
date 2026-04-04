import os
import io
import asyncio
from PIL import Image

from core.agent import Neo4jHistologyAgent
from utils.config import userdata

async def interactive_mode():
    """
    Initializes the Command Line Interface (CLI) for asynchronous terminal iteration.
    
    1. Grabs credentials.
    2. Inflates Neo4jHistologyAgent.
    3. Handles loops for chat memory and images.
    """
    print("""
============================================================
🔬 ASISTENTE DE HISTOLOGÍA - RAG MULTIMODAL GROQ (v4.4) 🔬
============================================================
Usando Llama-4-17B, LangGraph, UNI + PLIP, Neo4j, Qdrant
------------------------------------------------------------
Comandos:
  'salir'    -> Terminar la sesión
  'imagen <ruta>' -> Subir una imagen junto con tu próxima pregunta (Ej: imagen /tmp/foto.jpg)
============================================================
    """)
    
    neo4j_uri = userdata.get('NEO4J_URI') or "bolt://localhost:7687"
    neo4j_user = userdata.get('NEO4J_USERNAME') or "neo4j"
    neo4j_pass = userdata.get('NEO4J_PASSWORD') or "password"
    
    agent = Neo4jHistologyAgent(neo4j_uri, neo4j_user, neo4j_pass)
    await agent.initialize()
    
    chat_history = []
    active_image = None
    
    while True:
        try:
            if active_image:
                print("\n[📷 Imagen cargada y lista para la siguiente pregunta]")
            user_input = input("\n👤 Tu consulta: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
                
            if user_input.lower().startswith('imagen '):
                route = user_input[7:].strip()
                if os.path.exists(route):
                    try:
                        active_image = Image.open(route).convert('RGB')
                        print(f"✅ Imagen '{os.path.basename(route)}' cargada en memoria temporal.")
                        continue
                    except Exception as e:
                        print(f"❌ Error leyendo imagen: {e}")
                        continue
                else:
                    print(f"❌ La ruta no existe: {route}")
                    continue
            
            print("⏳ Analizando y procesando (LangGraph en curso)...")
            agent_result = await agent.query(user_input, session_id="cli_local", image=active_image, history=chat_history)
            
            print(f"\n🧠 ASISTENTE:\n{agent_result.get('answer', '')}\n")
            if agent_result.get("identified_structure"):
                print(f"   🏷️  Diagnóstico Visual: {agent_result['identified_structure']}")
            print("-" * 60)
            
            active_image = None
            
        except KeyboardInterrupt:
            print("\n👋  Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error inferencia: {e}")

    await agent.db.close()
