import os
import fitz # PyMuPDF
from PIL import Image
try:
    from pdf2image import convert_from_path
except ImportError:
    pass
import glob
try:
    import pytesseract
except ImportError:
    pass
from typing import List, Dict

from utils.config import DIRECTORIO_IMAGENES

class ExtractorImagenesPDF:
    """
    Clase encargada de leer archivos PDF y extraer sus imágenes, 
    guardándolas en disco junto con su metadato (OCR y contexto textual de la página).
    """
    MIN_WIDTH  = 200
    MIN_HEIGHT = 200

    def __init__(self, directorio_salida: str = DIRECTORIO_IMAGENES):
        """
        Inicializa el extractor de PDFs asegurando el directorio de salida.
        
        Args:
            directorio_salida (str): Ruta donde se guardarán las imágenes extraídas.
        """
        self.directorio_salida = directorio_salida
        os.makedirs(directorio_salida, exist_ok=True)

    def extraer_de_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Abre un PDF y extrae la imagen más relevante (por tamaño) de cada página.
        En caso de no encontrar imágenes extraíbles nativamente, realiza un renderizado (fallback)
        de la página completa. Extrae también texto por OCR y el texto de la página.

        Args:
            pdf_path (str): Ruta al archivo PDF.

        Returns:
            List[Dict[str, str]]: Lista de diccionarios con la metadata de cada imagen extraída 
                                  (ruta local, fuente, página, texto OCR y texto del manual).
        """
        imagenes_extraidas = []
        nombre_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error abriendo {pdf_path}: {e}")
            return []

        def _texto_pagina_con_contexto(doc, idx_0based: int) -> str:
            """
            Obtiene el texto de una página dada y la siguiente para propósitos de contexto (leyendas/captions).

            Args:
                doc (fitz.Document): Documento PDF abierto.
                idx_0based (int): Índice base cero de la página actual.

            Returns:
                str: Texto combinado de ambas páginas.
            """
            partes = []
            try:
                partes.append(doc[idx_0based].get_text().strip())
            except Exception:
                pass
            if idx_0based + 1 < len(doc):
                try:
                    siguiente = doc[idx_0based + 1].get_text().strip()
                    if siguiente:
                        partes.append(siguiente)
                except Exception:
                    pass
            return "\n".join(partes)

        for num_pagina, pagina in enumerate(doc, start=1):
            idx_0 = num_pagina - 1 
            valid_images_this_page = []
            
            try:
                img_info_list = pagina.get_image_info(xrefs=True)
                page_xrefs = [info["xref"] for info in img_info_list if info.get("xref")]
                
                if page_xrefs:
                    for xref in page_xrefs:
                        try:
                            base_image = doc.extract_image(xref)
                            if not base_image: continue
                            
                            image_bytes = base_image["image"]
                            from io import BytesIO
                            pil_temp = Image.open(BytesIO(image_bytes))
                            w, h = pil_temp.size
                            
                            if w >= self.MIN_WIDTH and h >= self.MIN_HEIGHT:
                                valid_images_this_page.append({
                                    "pil": pil_temp,
                                    "area": w * h
                                })
                        except Exception:
                            continue
            except Exception:
                pass
            
            if valid_images_this_page:
                mejor = max(valid_images_this_page, key=lambda x: x["area"])
                pil_img = mejor["pil"]
                nombre_archivo = f"{nombre_pdf}_pag{num_pagina}.png"
                ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
                
                try:
                    pil_img.save(ruta_completa, format="PNG")
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    
                    texto_completo_pagina = _texto_pagina_con_contexto(doc, idx_0)

                    imagenes_extraidas.append({
                        "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                        "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                        "texto_pagina": texto_completo_pagina
                    })
                except Exception as e:
                    print(f"  ⚠️ Error guardando pág {num_pagina}: {e}")
            else:
                try:
                    from pdf2image import convert_from_path
                    pag_imgs = convert_from_path(pdf_path, first_page=num_pagina, last_page=num_pagina, dpi=150)
                    if pag_imgs:
                        pil_full = pag_imgs[0]
                        nombre_archivo = f"{nombre_pdf}_pag{num_pagina}_full.png"
                        ruta_completa  = os.path.join(self.directorio_salida, nombre_archivo)
                        pil_full.save(ruta_completa, format="PNG")
                        
                        try:
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""

                        texto_completo_pagina = _texto_pagina_con_contexto(doc, idx_0)
                        
                        imagenes_extraidas.append({
                            "path": ruta_completa, "fuente_pdf": os.path.basename(pdf_path),
                            "pagina": num_pagina, "indice": 1, "ocr_text": ocr_text,
                            "texto_pagina": texto_completo_pagina
                        })
                except Exception as e:
                    print(f"  ⚠️ Fallback error pág {num_pagina}: {e}")

        doc.close()
        print(f"  📸 {len(imagenes_extraidas)} imágenes procesadas de {os.path.basename(pdf_path)}")
        return imagenes_extraidas

    def extraer_de_directorio(self, directorio: str) -> List[Dict[str, str]]:
        """
        Ejecuta la extracción de imágenes iterando por todos los PDFs dentro de un directorio.
        
        Args:
            directorio (str): Directorio que contiene archivos .pdf.

        Returns:
            List[Dict[str, str]]: Lista global de metadatos de imágenes extraídas.
        """
        todas = []
        pdfs  = glob.glob(os.path.join(directorio, "*.pdf"))
        print(f"📂 Extrayendo {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            todas.extend(self.extraer_de_pdf(pdf_path))
        print(f"✅ Total imágenes: {len(todas)}")
        return todas
