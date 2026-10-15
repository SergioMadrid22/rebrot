import requests
import json
from PIL import Image
import io
import base64
import binascii
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = f"http://127.0.0.1:{int(os.getenv('PORT_API', 7890))}"

def decode_image(hex_str):
    if not hex_str: return None
    data = binascii.unhexlify(hex_str)
    return Image.open(io.BytesIO(data))

def decode_image_b64(b64_str):
    """Decodifica string Base64 a PIL Image"""
    if not b64_str: return None
    try:
        data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(data))
    except Exception as e:
        print(f"Error decodificando imagen: {e}")
        return None

def detect_damage(image_path, confidence, dilatation, seg_model="yolo_sam"):
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "confidence": str(confidence), 
            "dilatation": str(dilatation),
            "seg_model": seg_model,
        }
        
        try:
            res = requests.post(f"{API_URL}/detect", files=files, data=data)
            
            # 1. Comprobar si el servidor devolvió un error HTTP (4xx o 5xx)
            if res.status_code != 200:
                # Intentar leer el detalle del error si es JSON
                try:
                    error_detail = res.json()
                    raise Exception(f"Error del Servidor ({res.status_code}): {error_detail}")
                except:
                    raise Exception(f"Error del Servidor ({res.status_code}): {res.text}")

            res_json = res.json()
            
            # 2. Comprobar errores lógicos controlados
            if res_json.get("status") == "error":
                raise Exception(res_json.get("message"))
                
            if res_json.get("status") == "no_damage":
                return None, None, [], None, None
            
            yolo_preview = decode_image_b64(res_json.get("yolo_preview"))
            mask = decode_image_b64(res_json["mask"])
            face_mask = decode_image_b64(res_json["face_mask"])
            raw_mask = decode_image_b64(res_json["raw_mask"])
            boxes = res_json.get("face_boxes", [])
            return yolo_preview, mask, boxes, face_mask, raw_mask
            
        except requests.exceptions.ConnectionError:
            raise Exception("No se puede conectar con el servidor de inferencia. ¿Está encendido?")
        except Exception as e:
            print(f"Error conexión API: {e}")
            raise e

def restore_image(image_path, mask_pil, face_mask_pil, raw_mask_pil, boxes, params):
    # Convertir PILs a bytes para enviar
    img_byte_arr = io.BytesIO()
    mask_byte_arr = io.BytesIO()
    
    mask_pil.save(mask_byte_arr, format='PNG')
    
    files = {
        'image': open(image_path, 'rb'),
        'mask': ('mask.png', mask_byte_arr.getvalue(), 'image/png'),
    }
    
    if face_mask_pil:
        fm_arr = io.BytesIO()
        face_mask_pil.save(fm_arr, format='PNG')
        files['face_mask'] = ('face.png', fm_arr.getvalue(), 'image/png')
        
    if raw_mask_pil:
        rm_arr = io.BytesIO()
        raw_mask_pil.save(rm_arr, format='PNG')
        files['raw_mask'] = ('raw.png', rm_arr.getvalue(), 'image/png')

    # Convertir params a form data
    data = params.copy()
    data['face_boxes_str'] = json.dumps(boxes) if boxes else "[]"
    data['model_variant'] = params.get('model_variant', 'sdxl')

    if 'use_controlnet' in data:
        data['use_controlnet'] = str(data['use_controlnet']).lower() 
    
    try:
        res = requests.post(f"{API_URL}/inpaint", files=files, data=data, timeout=600)
        
        if res.status_code == 200:
            # La respuesta es directamente la imagen binaria (bytes), no JSON
            return Image.open(io.BytesIO(res.content))
        else:
            try:
                err = res.json()
                msg = err.get("message", res.text)
            except:
                msg = res.text
            raise Exception(f"Fallo en Inpainting: {msg}")
            
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Error comunicando con API Inpainting: {e}")
    

def register_server_batch(count):
    try:
        requests.post(f"{API_URL}/queue/register_batch", data={"count": count}, timeout=2)
    except:
        pass

def get_server_queue():
    try:
        res = requests.get(f"{API_URL}/queue_status", timeout=2)
        if res.status_code == 200:
            return res.json()
    except:
        return {"queue_size": 0, "estimated_wait_seconds": 0, "status": "offline"}
    return {"queue_size": 0, "estimated_wait_seconds": 0, "status": "unknown"}
