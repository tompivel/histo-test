import os
import base64
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
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage
from utils.resilience import invoke_with_retry

from utils.config import IMG_DIR

class PDFImageExtractor:
    """
    Class responsible for reading PDF files and extracting their images, 
    saving them to disk along with their metadata (OCR and page contextual text).
    """
    MIN_WIDTH  = 200
    MIN_HEIGHT = 200

    def __init__(self, output_dir: str = IMG_DIR, llm=None):
        self.output_dir = output_dir
        self.llm = llm
        os.makedirs(output_dir, exist_ok=True)

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        extracted_images = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening {pdf_path}: {e}")
            return []

        def _page_text_with_context(doc, idx_0based: int) -> str:
            parts = []
            try:
                parts.append(doc[idx_0based].get_text().strip())
            except Exception:
                pass
            if idx_0based + 1 < len(doc):
                try:
                    next_page = doc[idx_0based + 1].get_text().strip()
                    if next_page:
                        parts.append(next_page)
                except Exception:
                    pass
            return "\n".join(parts)

        for page_num, page in enumerate(doc, start=1):
            idx_0 = page_num - 1 
            valid_images_this_page = []
            
            try:
                img_info_list = page.get_image_info(xrefs=True)
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
                best_img = max(valid_images_this_page, key=lambda x: x["area"])
                pil_img = best_img["pil"]
                file_name = f"{pdf_name}_pag{page_num}.png"
                full_path  = os.path.join(self.output_dir, file_name)
                
                try:
                    pil_img.save(full_path, format="PNG")
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()[:300]
                    except Exception:
                        ocr_text = ""
                    
                    full_page_text = _page_text_with_context(doc, idx_0)

                    extracted_images.append({
                        "path": full_path, "source_pdf": os.path.basename(pdf_path),
                        "page": page_num, "index": 1, "ocr_text": ocr_text,
                        "page_text": full_page_text
                    })
                except Exception as e:
                    print(f"  ⚠️ Error saving page {page_num}: {e}")
            else:
                try:
                    from pdf2image import convert_from_path
                    page_imgs = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)
                    if page_imgs:
                        pil_full = page_imgs[0]
                        file_name = f"{pdf_name}_pag{page_num}_full.png"
                        full_path  = os.path.join(self.output_dir, file_name)
                        pil_full.save(full_path, format="PNG")
                        
                        try:
                            ocr_text = pytesseract.image_to_string(pil_full).strip()[:300]
                        except Exception:
                            ocr_text = ""

                        full_page_text = _page_text_with_context(doc, idx_0)
                        
                        extracted_images.append({
                            "path": full_path, "source_pdf": os.path.basename(pdf_path),
                            "page": page_num, "index": 1, "ocr_text": ocr_text,
                            "page_text": full_page_text
                        })
                except Exception as e:
                    print(f"  ⚠️ Fallback error page {page_num}: {e}")

        doc.close()
        print(f"  📸 {len(extracted_images)} images processed from {os.path.basename(pdf_path)}")
        return extracted_images

    def extract_from_directory(self, directory: str) -> List[Dict[str, str]]:
        all_imgs = []
        pdfs  = glob.glob(os.path.join(directory, "*.pdf"))
        print(f"📂 Extracting from {len(pdfs)} PDFs...")
        for pdf_path in pdfs:
            all_imgs.extend(self.extract_from_pdf(pdf_path))
        print(f"✅ Total images extracted: {len(all_imgs)}")
        return all_imgs

    async def detect_and_extract_table(self, image_path: str) -> str:
        """Uses a multimodal LLM to detect and format tables in the page image.
        Attempts Groq Vision first, with fallback to Gemini 2.5 Flash.
        Returns Markdown table string or empty string if no table found."""
        if not self.llm:
            return ""
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            msg = HumanMessage(content=[
                {"type": "text", "text": (
                    "Analiza esta imagen de una página de un manual de histología.\n"
                    "Si encuentras tablas con datos técnicos, extráelas en formato Markdown.\n"
                    "Si no hay tablas, responde únicamente con 'SIN_TABLAS'."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
            ])

            # Attempt 1: Groq Vision
            try:
                from langchain_groq import ChatGroq
                from utils.config import userdata
                llm_vision = ChatGroq(
                    model="llama-3.2-11b-vision-preview",
                    api_key=userdata.get("GROQ_API_KEY"),
                    temperature=0, max_retries=1
                )
                resp = await invoke_with_retry(llm_vision, [msg])
                text = resp.content.strip()
                return "" if "SIN_TABLAS" in text else text
            except Exception as e_groq:
                print(f"  ⚠️ Groq Vision unavailable: {e_groq}")

            # Attempt 2: Fallback to Gemini 2.5 Flash
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from utils.config import userdata
                llm_gemini = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=userdata.get("GOOGLE_API_KEY"),
                    temperature=0,
                )
                resp = await invoke_with_retry(llm_gemini, [msg])
                text = resp.content.strip()
                return "" if "SIN_TABLAS" in text else text
            except Exception as e_gemini:
                print(f"  ⚠️ Gemini fallback also failed: {e_gemini}")
                return ""
        except Exception as e:
            print(f"  ⚠️ Table detection error: {e}")
            return ""
