import os
import io
import asyncio
from PIL import Image

from core.agent import AsistenteHistologiaNeo4j
from utils.config import userdata

async def modo_interactivo():
    """
    Inicia la Interfaz de Línea de Comandos (CLI) para interactuar con la consola asíncrona.
    
    Gestiona el bucle principal de evento donde:
    1. Se recuperan credenciales esenciales.
    2. Se infla el "AsistenteHistologiaNeo4j".
    3. Se evalúan comandos especiales como 'salir' o prefijos 'imagen /ruta' 
       mientras se acumula la memoria de contexto y se renderizan respuestas.
    """
    print("""
============================================================
🔬 ASISTENTE DE HISTOLOGÍA - RAG MULTIMODAL GROQ (v4.3) 🔬
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
    
    asistente = AsistenteHistologiaNeo4j(neo4j_uri, neo4j_user, neo4j_pass)
    await asistente.initialize()
    
    chat_history = []
    imagen_activa = None
    
    while True:
        try:
            if imagen_activa:
                print("\n[📷 Imagen cargada y lista para la siguiente pregunta]")
            user_input = input("\n👤 Tu consulta: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
                
            if user_input.lower().startswith('imagen '):
                rut = user_input[7:].strip()
                if os.path.exists(rut):
                    try:
                        imagen_activa = Image.open(rut).convert('RGB')
                        print(f"✅ Imagen '{os.path.basename(rut)}' cargada en memoria temporal.")
                        continue
                    except Exception as e:
                        print(f"❌ Error leyendo imagen: {e}")
                        continue
                else:
                    print(f"❌ La ruta no existe: {rut}")
                    continue
            
            print("⏳ Analizando y procesando (LangGraph en curso)...")
            respuesta = await asistente.consultar(user_input, imagen=imagen_activa, history=chat_history)
            
            print(f"\n🧠 ASISTENTE:\n{respuesta}\n")
            print("-" * 60)
            
            imagen_activa = None
            
        except KeyboardInterrupt:
            print("\n👋  Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error inferencia: {e}")

    await asistente.db.close()
