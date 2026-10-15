from dotenv import load_dotenv
load_dotenv()
import io
import os
import gc
import base64
import uuid
import json
import torch
import cv2
import numpy as np
from contextlib import contextmanager
from PIL import Image, ImageOps
import threading
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
import uvicorn
import traceback
import time
import logging

from libs.sam2.model import SAM2
from libs.langsam.model import LangSAMFaceExtractor
from libs.stable_diffusion.impaint.model import SDImpainting
from libs.yolov8.model import YOLOV8
from libs.unet.model import UNetInference
from libs.segformer.model import SegformerInference
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
from transformers.generation.configuration_utils import GenerationConfig
from transformers.configuration_utils import PretrainedConfig

# 1. Parche para get_head_mask
def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask

if not hasattr(transformers.modeling_utils.PreTrainedModel, "get_head_mask"):
    transformers.modeling_utils.PreTrainedModel.get_head_mask = get_head_mask

# 2. Parche para torch.Tensor.to (Error de dtype=device)
orig_tensor_to = torch.Tensor.to
def patched_tensor_to(self, *args, **kwargs):
    if "dtype" in kwargs and kwargs["dtype"] is not None:
        if not isinstance(kwargs["dtype"], torch.dtype):
            kwargs["device"] = kwargs.pop("dtype")
    return orig_tensor_to(self, *args, **kwargs)
torch.Tensor.to = patched_tensor_to

# 3. PARCHE PARA RESTA DE BOOLEANOS
orig_sub = torch.Tensor.__sub__
def patched_sub(self, other):
    if self.dtype == torch.bool:
        return orig_sub(self.to(torch.int8), other)
    return orig_sub(self, other)
torch.Tensor.__sub__ = patched_sub

orig_rsub = torch.Tensor.__rsub__
def patched_rsub(self, other):
    if self.dtype == torch.bool:
        # Esto maneja el caso (1.0 - tensor_booleano)
        return orig_rsub(self.to(torch.int8), other)
    return orig_rsub(self, other)
torch.Tensor.__rsub__ = patched_rsub

# 4. Parche para torch.finfo
orig_finfo = torch.finfo
def patched_finfo(dtype=None):
    if dtype is not None and not isinstance(dtype, torch.dtype):
        return orig_finfo(torch.get_default_dtype())
    return orig_finfo(dtype)
torch.finfo = patched_finfo

# 5. Parche para Qwen Config
orig_from_model_config = GenerationConfig.from_model_config
@classmethod
def patched_from_model_config(cls, model_config, **kwargs):
    for attr in ["text_config", "vision_config", "decoder_config"]:
        if hasattr(model_config, attr):
            sub_config = getattr(model_config, attr)
            if isinstance(sub_config, dict):
                setattr(model_config, attr, PretrainedConfig.from_dict(sub_config))
    return orig_from_model_config(model_config, **kwargs)
GenerationConfig.from_model_config = patched_from_model_config

LAST_ACTIVITY_TIME = time.time()
INACTIVITY_TIMEOUT = 60 * 15 # 15 minutos
DEVICE = os.environ.get("DEVICE", "cuda:0")
print(f"SERVIDOR: Usando dispositivo {DEVICE}")
YOLO_PATH = os.environ.get("YOLO_MODEL_PATH")
if not YOLO_PATH or not os.path.exists(YOLO_PATH):
    print(f"ADVERTENCIA: No se encontró el modelo YOLO en: {YOLO_PATH}")
    print("Por favor configura YOLO_MODEL_PATH en el archivo .env")
PORT_API = int(os.getenv("PORT_API", 7890))

gpu_lock = threading.Lock()

# Variables globales para la cola
active_requests = 0
pending_batch_tasks = 0
queue_lock = threading.Lock()
# Tiempos promedio iniciales (en segundos)
avg_times = {
    "detect": 5.0,
    "inpaint": 25.0
}
# Guardaremos los últimos N tiempos para suavizar la estimación
history_times = {"detect": [], "inpaint": []}

def update_avg_time(task_type, new_time):
    history_times[task_type].append(new_time)
    if len(history_times[task_type]) > 10:
        history_times[task_type].pop(0)
    avg_times[task_type] = sum(history_times[task_type]) / len(history_times[task_type])

def idle_model_cleaner():
    global LAST_ACTIVITY_TIME
    while True:
        time.sleep(30)
        elapsed = time.time() - LAST_ACTIVITY_TIME
        if elapsed > INACTIVITY_TIMEOUT:
            with gpu_lock:
                loaded_models = [m for m in models if models[m] is not None]
                if loaded_models:
                    print(f"Servidor inactivo por {int(elapsed/60)} minutos. Descargando modelos: {loaded_models}")
                    unload_models(to_unload=loaded_models)
                    LAST_ACTIVITY_TIME = time.time()

models = {
    "yolo": None,
    "sam": None,
    "unet": None,
    "segformer": None,
    "sdxl": None,
    "face": None,
    "flux": None,
    "qwen": None,
}

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.1)

def unload_models(to_unload: list):
    for m in to_unload:
        if m in models and models[m] is not None:
            print(f"SERVIDOR: Descargando {m}...")
            models[m] = None
    clear_vram()

def get_model(model_type, use_controlnet=False):
    global models
    
    if model_type in ["sdxl", "flux", "qwen"]:
        if model_type == "sdxl":
            unload_models(to_unload=["flux", "qwen"])
            if models["sdxl"] is None or models["sdxl"].use_controlnet != use_controlnet:
                print(f"SERVIDOR: Cargando SDXL (CN={use_controlnet})...")
                models["sdxl"] = SDImpainting(DEVICE, use_controlnet=use_controlnet)
            return models["sdxl"]
        
        elif model_type == "flux":
            unload_models(to_unload=["sdxl", "qwen"])
            if models["flux"] is None:
                print(f"SERVIDOR: Cargando Flux...")
                models["flux"] = FluxImpainting(DEVICE) 
            return models["flux"]

        elif model_type == "qwen":
            unload_models(to_unload=["sdxl", "flux", "face", "yolo", "sam", "unet", "segformer"])
            if models["qwen"] is None:
                print(f"SERVIDOR: Cargando Qwen...")
                models["qwen"] = QwenImpainting(DEVICE)
            return models["qwen"]

    if models.get(model_type) is None:
        unload_models(to_unload=["flux", "qwen"])
        print(f"SERVIDOR: Cargando {model_type} en {DEVICE}...")
        if model_type == "yolo":
            models["yolo"] = YOLOV8(device=DEVICE, model_path=YOLO_PATH)
        elif model_type == "sam":
            models["sam"] = SAM2(DEVICE)
        elif model_type == "face":
            models["face"] = LangSAMFaceExtractor(device=DEVICE)
        elif model_type == "unet":
            models["unet"] = UNetInference(DEVICE)
        elif model_type == "segformer":
            models["segformer"] = SegformerInference(DEVICE)
            
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
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img) 
    return img

app = FastAPI(title="Rebrot Inference Server")

# --- ENDPOINTS ---
@app.get("/queue_status")
def get_queue_status():
    with queue_lock:
        # El tiempo estimado es el tiempo promedio de inpaint por los que están en cola
        # (Es una simplificación, ya que no sabemos si son detect o inpaint)
        total = active_requests + pending_batch_tasks
        est_time = total * avg_times["inpaint"]
        return {
            "queue_size": total,
            "estimated_wait_seconds": round(est_time, 1),
            "status": "ocupado" if total > 0 else "libre"
        }

@app.post("/queue/register_batch")
def register_batch(count: int = Form(...)):
    global pending_batch_tasks
    with queue_lock:
        pending_batch_tasks += count
    return {"pending": pending_batch_tasks}

@app.post("/detect")
def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.1),
    dilatation: int = Form(10),
    seg_model: str = Form("yolo_sam")
):
    global active_requests, pending_batch_tasks, LAST_ACTIVITY_TIME
    LAST_ACTIVITY_TIME = time.time()
    with queue_lock:
        active_requests += 1
        # Si venía de un lote, lo restamos de pendientes porque ya es activa
        if pending_batch_tasks > 0: pending_batch_tasks -= 1

    start = time.time()

    with gpu_lock:
        try:
            contents = file.file.read()
            image_pil = read_image_file(contents)
            image_np = np.array(image_pil)

            with TempFileManager(image_pil, ".jpg") as temp_path:
                face = get_model("face")
                face_mask_np, face_boxes = face(temp_path, return_results="both", mask_multiplier=255)
                face_mask_np = fill_little_spaces(face_mask_np, 65)
            
                yolo_preview = None
                if seg_model == "yolo_sam":
                    yolo = get_model("yolo")
                    yolo_preview, boxes = yolo.get_bounding_box(confidence, temp_path)
                    if len(boxes) == 0:
                        update_avg_time("detect", time.time() - start)
                        return {"status": "no_damage", "mask": None}
                    sam = get_model("sam")
                    masks = sam.get_mask_by_bounding_boxes(boxes=boxes, image=image_np)
                    numpy_masks = [mask.cpu().numpy() for mask in masks]
                    combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
                    for m in numpy_masks:
                        combined_mask = np.logical_or(combined_mask, m)
                    binary_mask = generate_binary_mask(combined_mask)

                elif seg_model == "unet":
                    unet = get_model("unet")
                    binary_mask_rgb = unet.get_mask(image_np)
                    binary_mask = (cv2.cvtColor(binary_mask_rgb, cv2.COLOR_RGB2GRAY) > 127).astype(np.uint8) * 255
                
                elif seg_model == "segformer":
                    segformer = get_model("segformer")
                    binary_mask_rgb = segformer.get_mask(image_np)
                    binary_mask = (cv2.cvtColor(binary_mask_rgb, cv2.COLOR_RGB2GRAY) > 127).astype(np.uint8) * 255
                
                refined = delete_irrelevant_detected_pixels(binary_mask)
                refined = fill_little_spaces(refined)
                dilated = soften_contours(refined, dilatation)
                
                mask_damage = Image.fromarray(dilated, mode='L')
                if mask_damage.size != image_pil.size:
                    mask_damage = mask_damage.resize(image_pil.size, resample=Image.NEAREST)
                mask_face = Image.fromarray(face_mask_np).convert("L")
                
                mask_damage_np = np.array(mask_damage)
                mask_face_np = np.array(mask_face)
                
                final_mask_bool = (mask_damage_np == 255) & (mask_face_np != 255)
                final_mask_np = np.uint8(final_mask_bool) * 255

                final_mask_pil = Image.fromarray(final_mask_np)
                face_mask_pil = Image.fromarray(mask_face_np)
                raw_mask_pil = Image.fromarray(binary_mask)
                
                clear_vram()
                update_avg_time("detect", time.time() - start)
                return JSONResponse({
                    "status": "success",
                    "mask": img_to_base64(final_mask_pil),
                    "face_mask": img_to_base64(face_mask_pil),
                    "raw_mask": img_to_base64(raw_mask_pil),
                    "face_boxes": face_boxes.tolist() if face_boxes is not None else [],
                    "yolo_preview": img_to_base64(yolo_preview) if yolo_preview else None
                })

        except Exception as e:
            print(f"Server Error /detect: {e}")
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        finally:
            with queue_lock:
                active_requests -= 1

@app.post("/inpaint")
def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    face_mask: UploadFile = File(None),
    raw_mask: UploadFile = File(None),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
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
    global active_requests, pending_batch_tasks, LAST_ACTIVITY_TIME
    LAST_ACTIVITY_TIME = time.time()
    with queue_lock:
        active_requests += 1
        if pending_batch_tasks > 0: pending_batch_tasks -= 1
    
    start = time.time()

    with gpu_lock:
        try:
            img_pil = read_image_file(image.file.read())
            original_dpi = img_pil.info.get('dpi', (600, 600))
            print(f"SERVIDOR: DPI original de la imagen: {original_dpi}")
            exif_data = img_pil.info.get('exif', None)
            img_pil = img_pil.convert("RGB")
            mask_pil = Image.open(io.BytesIO(mask.file.read())).convert("L")

            inpainter = get_model(model_variant, use_controlnet=use_controlnet)
            padding_val = padding_crop if use_padding else None
            start_time = time.time()
            
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
                if face_mask and model_variant != "qwen":
                    f_mask_bytes = face_mask.file.read()
                    f_mask_pil = Image.open(io.BytesIO(f_mask_bytes)).convert("L")
                    f_mask_pil = ImageOps.autocontrast(f_mask_pil)

                    restored = restored.convert("RGBA")
                    orig_rgba = img_pil.convert("RGBA")
                    restored = Image.composite(orig_rgba, restored, f_mask_pil)
                    
                # 3. Enhance Faces
                print(f"SERVIDOR: Enhance Faces={enhance_faces}, Face Boxes={face_boxes_str}...")
                if enhance_faces and raw_mask and face_boxes_str != "[]" and model_variant != "qwen":
                    print("Servidor: Mejorando rostros...")
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
            
            try:
                if original_dpi and isinstance(original_dpi, (tuple, list)):
                    dpi_to_save = (int(round(original_dpi[0])), int(round(original_dpi[1])))
                else:
                    dpi_to_save = (600, 600)
            except:
                dpi_to_save = (600, 600)
            
            buf = io.BytesIO()
            
            save_params = {
                "format": "JPEG",
                "quality": 100,
                "dpi": dpi_to_save,
                "optimize": False
            }
            if exif_data is not None:
                save_params["exif"] = exif_data

            restored.convert("RGB").save(buf, **save_params)
            print(f"SERVIDOR: Imagen restaurada en {time.time() - start_time:.2f} segundos.")
            clear_vram()
            update_avg_time("inpaint", time.time() - start)
            return Response(content=buf.getvalue(), media_type="image/jpeg")

        except Exception as e:
            print(f"Server Error /inpaint: {e}")
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        
        finally:
            with queue_lock:
                active_requests -= 1

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Si el mensaje contiene "/queue_status" o "/health", devolvemos False para no mostrarlo
        message = record.getMessage()
        return message.find("/queue_status") == -1 and message.find("/health") == -1

# Aplicamos el filtro al logger de acceso de uvicorn
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

if __name__ == "__main__":
    cleaner_thread = threading.Thread(target=idle_model_cleaner, daemon=True)
    cleaner_thread.start()
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT_API,
        timeout_keep_alive=6000,
    )