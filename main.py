import asyncio
import nest_asyncio

# Aplica compatibilidad de event loops para Jupyter/Colab
nest_asyncio.apply()

from core.cli import modo_interactivo

if __name__ == "__main__":
    try:
        asyncio.run(modo_interactivo())
    except KeyboardInterrupt:
        print("\nSaliendo...")
