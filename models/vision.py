import torch
import numpy as np
from typing import Optional
from PIL import Image

import timm
from transformers import CLIPProcessor, CLIPModel

from utils.config import DIM_IMG_UNI, DIM_IMG_PLIP

class PlipWrapper:
    """
    Wrapper especializado del modelo CLIP de Visión Biomédica en Patología de Vincent (PLIP).
    Extrae representaciones vectoriales densas de imágenes (512 dimensiones) proyectadas al plano textual-visual biológico.
    """
    def __init__(self, device):
        """
        Inicializa variables del modelo.
        
        Args:
            device (str): Dispositivo de ejecución backend ("cuda" o "cpu").
        """
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """
        Instancia y carga en memoria el modelo HuggingFace `vinid/plip`, fijando su
        capa transaccional en modo evaluación (eval).
        """
        print("🔄 Cargando PLIP (vinid/plip)...")
        try:
            self.model = CLIPModel.from_pretrained("vinid/plip").to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained("vinid/plip")
            print("✅ PLIP cargado")
        except Exception as e:
            print(f"❌ Error cargando PLIP: {e}")

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Toma una imagen referenciada localmente y genera el sub-embedding respectivo.
        Funciona cortando explícitamente el cálculo de gradientes.

        Args:
            image_path (str): Ruta estática de la foto a evaluar (jpeg/png).

        Returns:
            np.ndarray: Vector 1-D de longitud 512, poblado con features semánticas.
        """
        if not self.model: return np.zeros(DIM_IMG_PLIP)
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                pooled = vision_out.pooler_output
                image_features = self.model.visual_projection(pooled)  # [1, 512]
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding PLIP: {e}")
            return np.zeros(DIM_IMG_PLIP)

class UniWrapper:
    """
    Wrapper de inferencia para Visual Transformer `MahmoodLab/UNI`, afinado para grandes volúmenes y alta resolución en secciones histológicas.
    Su dimensionalidad natural es de ViT-L/16 proyectada a 1024 floats.
    """
    def __init__(self, device):
        """
        Inicializa configuraciones básicas de pre-transformación.
        
        Args:
            device (str): Tipo de procesador (CUDA target).
        """
        self.device = device
        self.model = None
        self.transform = None

    def load(self):
        """
        Carga explícitamente usando la librería nativa para computación paralela de imágenes `timm`.
        La función resolve_data_config adapta la crop-size y la media/varianza a los tensores locales.
        """
        print("🔄 Cargando UNI (MahmoodLab)...")
        try:
            self.model = timm.create_model(
                "hf_hub:MahmoodLab/UNI", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=True
            )
            self.model.to(self.device).eval()
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
            self.transform = create_transform(**config)
            print("✅ UNI cargado")
        except Exception as e:
            print(f"❌ Error cargando UNI: {e}")

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Extrae un embedding puro (ViT Feature representation) usando configuraciones fijas.
        
        Args:
            image_path (str): Identificador en disco del archivo consultado.

        Returns:
            np.ndarray: Flatten array que encapsula las 1024 dimensionalidades finales extraídas.
        """
        if not self.model: return np.zeros(DIM_IMG_UNI)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = self.model(image_tensor) # UNI returns raw features [1, 1024]
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ Error embedding UNI: {e}")
            return np.zeros(DIM_IMG_UNI)
