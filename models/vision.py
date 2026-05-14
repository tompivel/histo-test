import torch
import numpy as np
from typing import Optional
from PIL import Image, ImageEnhance

def preprocess_image_for_embedding(image_path: str) -> Image.Image:
    """
    Preprocess USER image before generating embeddings.
    Applies only contrast/brightness enhancement — NO magnification.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Apply contrast enhancement (factor 1.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Apply brightness enhancement (factor 1.1)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        
        return img
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}, using original")
        return Image.open(image_path).convert('RGB')

import timm
from transformers import CLIPProcessor, CLIPModel

from utils.config import UNI_IMG_DIM, PLIP_IMG_DIM

class PlipWrapper:
    """
    Specialized wrapper for Vincent's Pathology Image (PLIP) Vision Transformer.
    Extracts dense vector representations natively aligned to biological textual semantics.
    """
    def __init__(self, device):
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        print("🔄 Loading PLIP (vinid/plip)...")
        try:
            self.model = CLIPModel.from_pretrained("vinid/plip").to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained("vinid/plip")
            print("✅ PLIP Loaded")
        except Exception as e:
            print(f"❌ Error loading PLIP: {e}")

    def embed_image(self, image_path: str, preprocess: bool = True) -> np.ndarray:
        if not self.model: return np.zeros(PLIP_IMG_DIM)
        try:
            if preprocess:
                image = preprocess_image_for_embedding(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                pooled = vision_out.pooler_output
                image_features = self.model.visual_projection(pooled)  # [1, 512]
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ PLIP embedding error: {e}")
            return np.zeros(PLIP_IMG_DIM)

class UniWrapper:
    """
    Inference wrapper for 'MahmoodLab/UNI' ViT engineered for histological sections.
    Projects image embeddings to 1024 dimensional features.
    """
    def __init__(self, device):
        self.device = device
        self.model = None
        self.transform = None

    def load(self):
        print("🔄 Loading UNI (MahmoodLab)...")
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
            print("✅ UNI Loaded")
        except Exception as e:
            print(f"❌ Error loading UNI: {e}")

    def embed_image(self, image_path: str, preprocess: bool = True) -> np.ndarray:
        if not self.model: return np.zeros(UNI_IMG_DIM)
        try:
            if preprocess:
                image = preprocess_image_for_embedding(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                emb = self.model(image_tensor) # UNI returns raw features [1, 1024]
            return emb.cpu().numpy().flatten()
        except Exception as e:
            print(f"⚠️ UNI embedding error: {e}")
            return np.zeros(UNI_IMG_DIM)
