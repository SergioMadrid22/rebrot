import io
import os
import gc
import base64
import uuid
import json
import torch
import cv2
from dotenv import load_dotenv
import numpy as np
from contextlib import contextmanager
from typing import List, Optional
from PIL import Image, ImageOps
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

from libs.sam2.model import SAM2
from libs.langsam.model import LangSAMFaceExtractor
from libs.stable_diffusion.impaint.model import SDImpainting
from libs.yolov8.model import YOLOV8
from libs.flux.model import FluxImpainting
from libs.qwen.model import QwenImpainting
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    crop_image
)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import transformers.modeling_utils

if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_head_mask"):
    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask
    
    transformers.modeling_utils.PreTrainedModel.get_head_mask = get_head_mask

    orig_tensor_to = torch.Tensor.to

def patched_tensor_to(*args, **kwargs):
    # Si la librería intenta pasar un objeto device en el lugar del dtype, lo movemos a 'device'
    if "dtype" in kwargs and isinstance(kwargs["dtype"], torch.device):
        kwargs["device"] = kwargs.pop("dtype")
    return orig_tensor_to(*args, **kwargs)

torch.Tensor.to = patched_tensor_to

orig_sub = torch.Tensor.__sub__
def patched_sub(self, other):
    if self.dtype == torch.bool:
        # Si es una resta de booleanos (común en GroundingDINO), convertimos a entero
        return orig_sub(self.to(torch.int8), other).to(torch.bool)
    return orig_sub(self, other)

# También parcheamos la resta desde el lado derecho (rsub) por si acaso
orig_rsub = torch.Tensor.__rsub__
def patched_rsub(self, other):
    if self.dtype == torch.bool:
        return orig_rsub(self.to(torch.int8), other).to(torch.bool)
    return orig_rsub(self, other)

torch.Tensor.__sub__ = patched_sub
torch.Tensor.__rsub__ = patched_rsub

orig_finfo = torch.finfo
def patched_finfo(dtype=None):
    # Si la librería pasa un device en lugar de un dtype, usamos el dtype por defecto
    if dtype is not None and not isinstance(dtype, torch.dtype):
        return orig_finfo(torch.get_default_dtype())
    return orig_finfo(dtype)
torch.finfo = patched_finfo

# Configuración
load_dotenv()
DEVICE = "cuda:0"#os.environ.get("CUDA_DEVICE", "cuda:1")
DEVICE2 = "cuda:1" if DEVICE == "cuda:0" else "cuda:0"
print(f"⚙️ SERVIDOR: Usando dispositivo {DEVICE}")
YOLO_PATH = os.environ.get("YOLO_MODEL_PATH")
if not YOLO_PATH or not os.path.exists(YOLO_PATH):
    print(f"⚠️ ADVERTENCIA: No se encontró el modelo YOLO en: {YOLO_PATH}")
    print("Por favor configura YOLO_MODEL_PATH en el archivo .env")

gpu_lock = threading.Lock()

models = {
    "yolo": None,
    "sam": None,
    "sdxl": None,
    "face": None,
    "flux": None,
    "qwen": None,
}

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()

def unload_heavy_models(except_model=None):
    """Descarga modelos pesados excepto el solicitado"""
    heavy = ["sdxl", "flux", "qwen"]
    for m in heavy:
        if m != except_model and models.get(m) is not None:
            print(f"SERVIDOR: Descargando {m} para liberar VRAM...")
            models[m] = None
    clear_vram()

def get_model(model_type, use_controlnet=False):
    global models
    
    # --- GESTIÓN DE MODELOS PESADOS ---
    if model_type in ["sdxl", "flux", "qwen"]:
        unload_heavy_models(except_model=model_type)
        
        if model_type == "sdxl":
             if models["sdxl"] is None or models["sdxl"].use_controlnet != use_controlnet:
                print(f"SERVIDOR: Cargando SDXL (CN={use_controlnet})...")
                models["sdxl"] = SDImpainting(DEVICE, use_controlnet=use_controlnet)
             return models["sdxl"]
        
        elif model_type == "flux":
            if models["flux"] is None:
                print(f"SERVIDOR: Cargando Flux...")
                models["flux"] = FluxImpainting(DEVICE2) 
            return models["flux"]

        elif model_type == "qwen":
            if models["qwen"] is None:
                print(f"SERVIDOR: Cargando Qwen...")
                models["qwen"] = QwenImpainting(DEVICE2)
            return models["qwen"]

    # --- MODELOS LIGEROS ---
    if models.get(model_type) is None:
        print(f"SERVIDOR: Cargando {model_type} en {DEVICE}...")
        if model_type == "yolo":
            models["yolo"] = YOLOV8(device=DEVICE, model_path=YOLO_PATH)
        elif model_type == "sam":
            models["sam"] = SAM2(DEVICE)
        elif model_type == "face":
            models["face"] = LangSAMFaceExtractor(device=DEVICE)
            
    return models[model_type]

# --- UTILIDADES GESTIÓN DE ARCHIVOS ---
@contextmanager
def TempFileManager(image_pil: Image.Image, suffix=".png"):
    """
    Context manager para guardar una imagen temporalmente con un nombre único (UUID)
    y borrarla automáticamente al salir del bloque 'with'.
    """
    unique_name = f"temp_{uuid.uuid4()}{suffix}"
    try:
        image_pil.save(unique_name)
        yield unique_name
    finally:
        if os.path.exists(unique_name):
            try:
                os.remove(unique_name)
            except Exception as e:
                print(f"Error borrando temp {unique_name}: {e}")

def img_to_base64(image_pil: Image.Image) -> str:
    """Convierte imagen PIL a string Base64 UTF-8"""
    if image_pil is None: return None
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def read_image_file(file_bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


app = FastAPI(title="Rebrot Inference Server")

# --- ENDPOINTS ---
@app.post("/detect")
def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.1),
    dilatation: int = Form(10)
):
    with gpu_lock:
        try:
            # 1. Leer imagen
            contents = file.file.read()
            image_pil = read_image_file(contents)
            image_np = np.array(image_pil)
            
            with TempFileManager(image_pil, ".jpg") as temp_path:
                # 2. YOLO
                yolo = get_model("yolo")
                yolo_preview, boxes = yolo.get_bounding_box(confidence, temp_path)
                
                if len(boxes) == 0:
                    return {"status": "no_damage", "mask": None}

                # 3. Face Detection
                face = get_model("face")
                face_mask_np, face_boxes = face(temp_path, return_results="both", mask_multiplier=255)
                face_mask_np = fill_little_spaces(face_mask_np, 65)
                
                boxes_list = []
                if face_boxes is not None:
                    if hasattr(face_boxes, "tolist"):
                        boxes_list = face_boxes.tolist()
                    else:
                        boxes_list = list(face_boxes)
                
                # 4. SAM
                sam = get_model("sam")
                masks = sam.get_mask_by_bounding_boxes(boxes=boxes, image=image_np)
                
                # Combinar máscaras SAM
                numpy_masks = [mask.cpu().numpy() for mask in masks]
                combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
                for m in numpy_masks:
                    combined_mask = np.logical_or(combined_mask, m)
                    
                binary_mask = generate_binary_mask(combined_mask)
                
                # 5. Refinamiento
                refined = delete_irrelevant_detected_pixels(binary_mask)
                refined = fill_little_spaces(refined)
                dilated = soften_contours(refined, dilatation)
                
                # 6. Restar Caras
                mask_damage = Image.fromarray(dilated, mode='L')
                mask_face = Image.fromarray(face_mask_np).convert("L")
                
                mask_damage_np = np.array(mask_damage)
                mask_face_np = np.array(mask_face)
                
                final_mask_bool = (mask_damage_np == 255) & (mask_face_np != 255)
                final_mask_np = np.uint8(final_mask_bool) * 255

                final_mask_pil = Image.fromarray(final_mask_np)
                face_mask_pil = Image.fromarray(mask_face_np)
                raw_mask_pil = Image.fromarray(binary_mask)
                
                # Retornar imágenes codificadas
                return JSONResponse({
                    "status": "success",
                    "mask": img_to_base64(final_mask_pil),
                    "face_mask": img_to_base64(face_mask_pil),
                    "raw_mask": img_to_base64(raw_mask_pil),
                    "face_boxes": boxes_list,
                    "yolo_preview": img_to_base64(yolo_preview) if yolo_preview else None
                })

        except Exception as e:
            print(f"❌ Server Error /detect: {e}")
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/inpaint")
def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    face_mask: UploadFile = File(None),
    raw_mask: UploadFile = File(None),
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
    strength: float = Form(0.99),
    guidance: float = Form(7.5),
    steps: int = Form(20),
    use_padding: bool = Form(False),
    padding_crop: int = Form(32),
    enhance_faces: bool = Form(True),
    face_boxes_str: str = Form("[]"),
    use_controlnet: bool = Form(False),
    model_variant: str = Form("sdxl"),
):
    with gpu_lock:
        try:
            img_pil = read_image_file(image.file.read())
            mask_pil = Image.open(io.BytesIO(mask.file.read())).convert("L")

            inpainter = get_model(model_variant, use_controlnet=use_controlnet)
            padding_val = padding_crop if use_padding else None
            
            with TempFileManager(img_pil, ".png") as t_img_path, \
                TempFileManager(mask_pil, ".png") as t_mask_path:
                
                # 1. Inpainting Principal
                main_kwargs = {
                    "image_path": t_img_path,
                    "mask_path": t_mask_path,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": strength,
                    "guidance": guidance,
                    "steps": steps,
                }

                if model_variant == "sdxl":
                    main_kwargs["padding_mask_crop"] = padding_val
                    main_kwargs["controlnet_scale"] = 0.5 if use_controlnet else None

                restored = inpainter.impaint(**main_kwargs)
                
                # 2. Componer Rostros Originales (Anti-alucinación)
                if face_mask:
                    f_mask_bytes = face_mask.file.read()
                    f_mask_pil = Image.open(io.BytesIO(f_mask_bytes)).convert("L")
                    f_mask_pil = ImageOps.autocontrast(f_mask_pil)

                    restored = restored.convert("RGBA")
                    orig_rgba = img_pil.convert("RGBA")
                    restored = Image.composite(orig_rgba, restored, f_mask_pil)
                    
                # 3. Enhance Faces
                print(f"✨ SERVIDOR: Enhance Faces={enhance_faces}, Face Boxes={face_boxes_str}...")
                if enhance_faces and raw_mask and face_boxes_str != "[]" and model_variant != "qwen":
                    print("✨ Servidor: Mejorando rostros...")
                    face_boxes = json.loads(face_boxes_str)
                    raw_mask_bytes = raw_mask.file.read()
                    raw_mask_pil = Image.open(io.BytesIO(raw_mask_bytes)).convert("L")
                    raw_mask_np = np.array(raw_mask_pil)
                    
                    image_bgr = cv2.imread(t_img_path)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    padding = 10
                    
                    # Requerimos el detector de caras para generar máscaras suaves en los recortes
                    face_detector_model = get_model("face") 

                    for box in face_boxes:
                        xmax, xmin, ymax, ymin = map(int, box)
                        x1 = max(0, xmin - padding)
                        y1 = max(0, ymin - padding)
                        x2 = min(image_rgb.shape[1], xmax + padding)
                        y2 = min(image_rgb.shape[0], ymax + padding)
                        
                        face_crop = crop_image(image_rgb, x1, y1, x2, y2)
                        face_mask_crop = crop_image(raw_mask_np, x1, y1, x2, y2)
                        
                        # Si hay daño en la cara
                        print(f"Revisando cara en box {box} con daño en máscara: {np.any(face_mask_crop == 255)}")
                        if np.any(face_mask_crop == 255):
                            with TempFileManager(Image.fromarray(face_crop), ".png") as t_fc, \
                                TempFileManager(Image.fromarray(face_mask_crop), ".png") as t_fm:

                                face_kwargs = {
                                    "image_path": t_fc,
                                    "mask_path": t_fm,
                                    "prompt": "clear skin, smooth face, realistic, natural lighting, photo restoration, high detail, close-up portrait",
                                    "negative_prompt": "scars, acne, dirty texture, unnatural lighting, distortions, plastic look",
                                    "strength": strength,
                                    "guidance": guidance,
                                    "steps": steps
                                }

                                if model_variant == "sdxl":
                                    face_kwargs["padding_mask_crop"] = None
                            
                                enhanced_face = inpainter.impaint(**face_kwargs)
                                
                                # Generar mascara alpha para pegado suave
                                enh_mask_np, _ = face_detector_model(t_fc, return_results="both", mask_multiplier=255)
                                enh_mask_np = fill_little_spaces(enh_mask_np, 65)
                                
                                enhanced_face = enhanced_face.convert("RGBA")
                                mask_arr = np.array(enh_mask_np)
                                rgba_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
                                rgba_arr[..., :3] = np.array(enhanced_face)[..., :3]
                                rgba_arr[..., 3] = mask_arr
                                
                                final_paste = Image.fromarray(rgba_arr, mode="RGBA")
                                restored.paste(final_paste, (x2, y1-enhanced_face.size[1]), final_paste)
            
            buf = io.BytesIO()
            restored.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")

        except Exception as e:
            print(f"❌ Server Error /inpaint: {e}")
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7890)