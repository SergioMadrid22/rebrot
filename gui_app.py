import gradio as gr
import os
import tempfile
import shutil
import re
from PIL import Image, ImageOps
from api_client import detect_damage, restore_image, get_server_queue, register_server_batch
from datetime import datetime
import traceback
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import time
import threading
background_tasks = {}

load_dotenv()

BASE_OUTPUT_DIR = "outputs/batches"
PORT_GUI = int(os.getenv("PORT_GUI", 7860))
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

DILATION_DEFAULTS = {
    "yolo_sam": 100,
    "unet": 50,
    "segformer": 90
}

def update_manual_dilation(model_name):
    # Devuelve el valor del diccionario, si no existe usa 100 por defecto
    return DILATION_DEFAULTS.get(model_name, 100)

def toggle_batch_dilation(use_default):
    return gr.update(interactive=not use_default)

def update_prompts_by_model(model_name):
    """Cambia los prompts manuales y los pasos según el modelo seleccionado."""
    if model_name == "qwen":
        return PROMPT_QWEN, NEG_QWEN, 8
    else:
        return PROMPT_SDXL, NEG_SDXL, 20

def get_user_folders(username):
    """Devuelve la lista de nombres de carpetas procesadas por el usuario."""
    if not username: return []
    user_dir = os.path.join(BASE_OUTPUT_DIR, username)
    if not os.path.exists(user_dir): return []
    
    # Listar solo directorios (excluyendo los archivos .zip)
    folders = [d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))]
    # Ordenar por fecha de creación (más reciente primero)
    folders.sort(key=lambda x: os.path.getmtime(os.path.join(user_dir, x)), reverse=True)
    return folders

def load_images_from_folder(username, folder_name):
    """Devuelve una lista de rutas de imágenes dentro de una carpeta y su ZIP asociado."""
    if not username or not folder_name:
        return [], None
    
    base_path = os.path.join(BASE_OUTPUT_DIR, username, folder_name)
    image_list = []
    
    # Recorrer subcarpetas (qwen, sdxl_yolo, etc.) buscando imágenes
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_list.append(os.path.join(root, file))
    
    # Buscar el ZIP correspondiente
    zip_path = os.path.join(BASE_OUTPUT_DIR, username, f"{folder_name}.zip")
    if not os.path.exists(zip_path):
        zip_path = None
        
    return image_list, zip_path

def send_completion_email(recipient_email, username, zip_name):
    """Envía un correo notificando que el procesamiento ha terminado."""
    if not recipient_email or "@" not in recipient_email:
        return False
    
    try:
        msg = EmailMessage()
        msg.set_content(f"""
Hola {username},

Te informamos que el procesamiento por lotes del Proyecto Rebrot ha finalizado con éxito.

Archivo generado: {zip_name}

Ya puedes descargar los resultados desde la pestaña 'Mis Descargas' en la aplicación.

Atentamente,
El equipo de Proyecto Rebrot
        """)

        msg['Subject'] = 'Procesamiento Rebrot Finalizado'
        msg['From'] = os.getenv("SENDER_EMAIL")
        msg['To'] = recipient_email

        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"❌ Error enviando email: {e}")
        traceback.print_exc()
        return False

def save_temp_image(image_pil, suffix=".jpg"):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image_pil.save(t.name)
    return t.name

def reset_state():
    """Limpia todos los componentes y estados dependientes de la imagen anterior."""
    return (
        None,  # im_yolo
        None,  # im_mask_editor
        "Sube una imagen y pulsa Detectar para comenzar.", # lbl_stat
        None,  # st_boxes (State)
        None,  # st_f_mask (State)
        None,  # st_r_mask (State)
        None,  # im_comparison (Slider)
        None,  # im_side_before
        None   # im_side_after
    )

# --- SISTEMA DE SESIONES ---
def login(username):
    """Valida el usuario, crea su directorio y cambia a la vista principal."""
    clean_user = re.sub(r'\W+', '_', username.strip().lower())
    if not clean_user:
        return (
            gr.update(visible=True), 
            gr.update(visible=False), 
            "", 
            "<span style='color:red'>❌ Nombre no válido. Usa letras y números.</span>", # Texto en rojo aquí
            ""
        )
    
    # Crear carpeta permanente del usuario si no existe
    user_dir = os.path.join(BASE_OUTPUT_DIR, clean_user)
    os.makedirs(user_dir, exist_ok=True)
    
    return (
        gr.update(visible=False), 
        gr.update(visible=True), 
        clean_user, 
        "", 
        f"### 👤 Usuario activo: **{clean_user}**"
    )

def refresh_history(username):
    """Lee la carpeta del usuario y devuelve la lista de archivos ZIP generados."""
    if not username: return None
    user_dir = os.path.join(BASE_OUTPUT_DIR, username)
    if not os.path.exists(user_dir): return None
    
    # Obtener archivos ZIP y ordenarlos del más reciente al más antiguo
    files = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f.endswith(".zip")]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


# --- LÓGICA MANUAL ---
def ui_detect(img_path, conf, dilat, seg_model):
    if not img_path: 
        return None, None, "Falta imagen", None, None, None
    try:
        yolo_prev, mask, boxes, f_mask, r_mask = detect_damage(img_path, conf, dilat, seg_model)
        
        if mask is None: 
            return None, None, "No se detectó daño", None, None, None
        
        original_img = Image.open(img_path)
        original_img = ImageOps.exif_transpose(original_img).convert("RGB")
        mask_overlay = Image.new("RGBA", original_img.size, (255, 255, 255, 160))
        mask_overlay.putalpha(mask.convert("L"))
        
        combined_editor_value = {
            "background": original_img,
            "layers": [mask_overlay],
            "composite": None
        }
        
        return yolo_prev, combined_editor_value, "✅ Daño detectado. Puedes pintar o borrar sobre la imagen.", boxes, f_mask, r_mask
    except Exception as e:
        return None, None, f"Error: {e}", None, None, None

def ui_restore(img_path, editor_data, prompt, neg, strn, guid, steps, pad, pad_sz, enh, model_var, state_boxes, state_f_mask, state_r_mask, state_history):
    if not img_path: 
        return None, None, None
    
    if model_var == "qwen" and (not editor_data or editor_data.get('background') is None):
        bg = Image.open(img_path).convert("RGB")
        mask_pil = Image.new("L", bg.size, 255)
    else:
        if not editor_data: return None, None, None
        
        if editor_data.get('layers') and len(editor_data) > 0:
            mask_pil = editor_data["layers"][0].convert("L")
            mask_pil = mask_pil.point(lambda p: 255 if p > 0 else 0)
        else:
            bg = editor_data
            mask_pil = Image.new("L", bg.size, 0)
    
    params = {
        "prompt": prompt, "negative_prompt": neg, "strength": strn,
        "guidance": guid, "steps": steps, "use_padding": pad,
        "padding_crop": pad_sz, "enhance_faces": enh,
        "use_controlnet": False, "model_variant": model_var,
    }
    
    try:
        res_img = restore_image(img_path, mask_pil, state_f_mask, state_r_mask, state_boxes, params)
        out_path = save_temp_image(res_img)

        new_history = [out_path] + state_history
        new_history = new_history[:10]

        return (img_path, out_path), img_path, out_path, new_history, new_history
    except Exception as e:
        raise gr.Error(f"Error Inpainting: {e}")


def run_batch_worker(
    task_id, files, conf, dilat, enh, use_default_dilat, model_vars,
    seg_models, configs, username, email_notification, batch_name
):
    try:
        clean_batch_name = re.sub(r'\W+', '_', batch_name.strip().lower())
        if not clean_batch_name:
            clean_batch_name = "batch"
        parts = task_id.split("_")
        timestamp = f"{parts[-2]}_{parts[-1]}"        
        user_dir = os.path.join(BASE_OUTPUT_DIR, username)
        folder_identifier = f"{clean_batch_name}_{timestamp}"
        batch_dir = os.path.join(user_dir, folder_identifier)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Filtramos modelos que SI necesitan máscara
        masked_models = [m for m in model_vars if m != "qwen"]
        qwen_selected = "qwen" in model_vars

        # Cálculo de tareas totales para el log
        total = (len(files) if qwen_selected else 0) + (len(files) * len(seg_models) * len(masked_models))
        count = 0
        
        # --- FASE 1: PROCESAMIENTO QWEN (Una sola vez por imagen) ---
        if qwen_selected:
            model_subdir = os.path.join(batch_dir, "qwen")
            os.makedirs(model_subdir, exist_ok=True)
            current_cfg = configs["qwen"]
            
            for f in files:
                count += 1
                file_path = f if isinstance(f, str) else f.name
                background_tasks[task_id] = f"Procesando {count}/{total} (Qwen - Sin máscara)"
                
                try:
                    # Qwen no necesita segmentación, creamos máscara blanca ficticia
                    img_temp = Image.open(file_path)
                    dummy_mask = Image.new("L", img_temp.size, 255)
                    
                    params = {
                        "prompt": current_cfg["prompt"],
                        "negative_prompt": current_cfg.get("neg", ""),
                        "strength": 1.0, # Qwen ignora strength usualmente
                        "steps": current_cfg.get("steps", 20),
                        "model_variant": "qwen",
                        "enhance_faces": False # Qwen no usa el pipeline de rostros local
                    }
                    
                    res_img = restore_image(file_path, dummy_mask, None, None, [], params)
                    
                    fname = os.path.basename(file_path)
                    name_no_ext = os.path.splitext(fname)[0]
                    res_img.convert("RGB").save(
                        os.path.join(model_subdir, f"qwen_{name_no_ext}.jpg"),
                        format="JPEG",
                        quality=100
                    )
                except Exception as e:
                    print(f"Error en Qwen batch: {e}")

        # --- FASE 2: PROCESAMIENTO CON MÁSCARA (SDXL / Flux) ---
        if masked_models:
            for seg_m in seg_models:
                current_dil = DILATION_DEFAULTS.get(seg_m, dilat) if use_default_dilat else dilat
                for f in files:
                    file_path = f if isinstance(f, str) else f.name
                    
                    try:
                        # 1. Detectamos daño una sola vez para este modelo de segmentación
                        background_tasks[task_id] = f"Segmentando con {seg_m}: {os.path.basename(file_path)}"
                        _, mask, boxes, f_mask, r_mask = detect_damage(file_path, conf, current_dil, seg_m)
                        
                        if mask is None: continue 

                        # 2. Restauramos con cada modelo de inpainting seleccionado (excepto qwen)
                        for model_var in masked_models:
                            count += 1
                            background_tasks[task_id] = f"Procesando {count}/{total} ({seg_m} + {model_var})"
                            
                            model_subdir = os.path.join(batch_dir, f"{seg_m}_{model_var}")
                            os.makedirs(model_subdir, exist_ok=True)
                            current_cfg = configs[model_var]

                            params = {
                                "prompt": current_cfg["prompt"],
                                "negative_prompt": current_cfg.get("neg", ""),
                                "strength": current_cfg.get("str", 0.99),
                                "guidance": current_cfg.get("guid", 7.5),
                                "steps": current_cfg.get("steps", 20),
                                "use_padding": False, 
                                "enhance_faces": enh, 
                                "model_variant": model_var
                            }
                            
                            res_img = restore_image(file_path, mask, f_mask, r_mask, boxes, params)
                            
                            fname = os.path.basename(file_path)
                            name_no_ext = os.path.splitext(fname)[0]
                            res_img.convert("RGB").save(
                                os.path.join(model_subdir, f"restored_{name_no_ext}.jpg"),
                                format="JPEG",
                                quality=100
                            )
                    except Exception as e:
                        print(f"Error en bucle de máscara: {e}")

        # Finalizing: ZIP and Cleanup
        background_tasks[task_id] = "Empaquetando resultados..."
        zip_path_base = os.path.join(user_dir, folder_identifier)
        full_zip_path = shutil.make_archive(zip_path_base, 'zip', batch_dir)
        
        # Cleanup the folder after zipping
        #shutil.rmtree(batch_dir)
        
        if email_notification:
            send_completion_email(email_notification, username, os.path.basename(full_zip_path))
            
        background_tasks[task_id] = "COMPLETADO"
    except Exception as e:
        background_tasks[task_id] = f"ERROR: {str(e)}"
        traceback.print_exc()

def batch_process(
    files, conf, dilat, enh, use_default_dilat, model_vars, seg_models,
    p_sdxl, n_sdxl, s_sdxl, g_sdxl, st_sdxl,
    p_flux, n_flux, s_flux, g_flux, st_flux,
    p_qwen, n_qwen, st_qwen,
    username, email_notification,
    batch_name,
    progress=gr.Progress()
):
    if not files or not model_vars: return None, "Error: No hay archivos"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"{username}_{timestamp}"
    background_tasks[task_id] = "Iniciando..."
    
    masked_models = [m for m in model_vars if m != "qwen"]
    qwen_selected = "qwen" in model_vars
    total_tasks = (len(files) if qwen_selected else 0) + (len(files) * len(seg_models) * len(masked_models))
    register_server_batch(total_tasks)
    
    # 2. Prepare Configs
    configs = {
        "sdxl": {"prompt": p_sdxl, "neg": n_sdxl, "str": s_sdxl, "guid": g_sdxl, "steps": st_sdxl},
        "flux": {"prompt": p_flux, "neg": n_flux, "str": s_flux, "guid": g_flux, "steps": st_flux},
        "qwen": {"prompt": p_qwen, "neg": n_qwen, "steps": st_qwen}
    }

    # 3. START BACKGROUND THREAD
    thread = threading.Thread(
        target=run_batch_worker,
        args=(task_id, files, conf, dilat, enh, use_default_dilat, 
              model_vars, seg_models, configs, 
              username, email_notification, batch_name)
    )
    thread.start()

    # 4. MONITOR (Yield progress while the session is open)
    while True:
        status = background_tasks.get(task_id, "Iniciando...")
        if status == "COMPLETADO":
            clean_batch_name = re.sub(r'\W+', '_', batch_name.strip().lower()) or "batch"
            zip_file = os.path.join(BASE_OUTPUT_DIR, username, f"{clean_batch_name}_{timestamp}.zip")
            yield zip_file, "✅ ¡Finalizado con éxito!"
            break
        elif "ERROR" in status:
            yield None, f"❌ {status}"
            break
        
        yield None, f"{status}"
        time.sleep(2)

def update_queue_display():
    status = get_server_queue()
    
    if status["status"] == "offline":
        return "🔴 **Servidor Desconectado**", "---"
    
    size = status["queue_size"]
    wait = status["estimated_wait_seconds"]
    
    if wait > 60:
        wait_str = f"{int(wait // 60)}m {int(wait % 60)}s"
    else:
        wait_str = f"{wait}s"
        
    icon = "🟡" if size > 0 else "🟢"
    msg = f"{icon} **Servidor {status['status']}** | Imágenes en cola: **{size}**"
    time_msg = f"⏳ T. espera estimado: **{wait_str}**"
    
    return msg, time_msg


# --- INTERFAZ ---
HELP_CONTENT = """
# Guía de Reparación Digital de Fotografías - Proyecto Rebrot

Esta herramienta utiliza Inteligencia Artificial avanzada para recuperar fotografías dañadas por la DANA. Para obtener los mejores resultados, es fundamental seleccionar el modelo de segmentación e inpainting adecuado según el tipo y gravedad del daño.

---

## Modelos de Reparación (Inpainting / Generación)

### 1. SDXL (Stable Diffusion XL 1.0)
* **Mejor para:** Manchas localizadas, fondos, ropa, grietas y bordes erosionados.
* **Flujo:** Requiere máscara (automática o manual). La IA solo actúa dentro de la zona delimitada, dejando el resto de la fotografía 100% idéntico al original.
* **Configuración recomendada:**
  * **Pasos (Steps):** 20 — 25
  * **Fidelidad al Prompt (Guidance):** 7.0 — 8.0
  * **Fuerza (Strength):** 0.99

### 2. Qwen-Image-Edit (Acelerado con Lightning LoRA)
* **Mejor para:** Daños extensos, manchas semi-transparentes o cuando el barro tapa elementos complejos de la escena.
* **Flujo:** No requiere máscara de daño; reinterpreta y limpia la escena de forma semántica y global mediante instrucciones de texto.
* **Optimización integrada:** Gracias al adaptador Lightning LoRA, **solo necesita entre 4 y 8 pasos** (por defecto: 8), reduciendo el tiempo de procesamiento drásticamente.
* **Configuración recomendada:**
  * **Pasos (Steps):** 6 — 8
  * **Fidelidad (Guidance):** 1.2 — 2.0

### 3. FLUX.1-Fill-dev
* **Mejor para:** Casos complejos donde SDXL no logre integrar bien la textura.
* **Inconvenientes:** Alto consumo de memoria y mayor tiempo de cómputo por imagen.
* **Configuración recomendada:**
  * **Pasos (Steps):** 20 — 25
  * **Fidelidad (Guidance):** 3.5 — 4.5

---

## Modelos de Detección y Segmentación de Daños

* **SegFormer (Recomendado por defecto):** Modelo basado en Transformers. Es el más rápido en inferencia y el mejor valorado en los benchmarks para delimitar bordes de manchas de barro de forma suave y precisa.
* **YOLO + SAM:** Combina detección rápida de manchas con segmentación fina de SAM2. Excelente para aislar daños individuales bien delimitados.
* **U-Net:** Red clásica de segmentación. Útil para daños continuos, aunque tiende a generar contornos ligeramente más holgados.

---

## Parámetros Clave

### Detección y Máscaras
* **Confianza Detección (YOLO):** Define la sensibilidad ante manchas sutiles. Si hay barro tenue que la IA no detecta, baja este valor (ej. 0.05).
* **Dilatación de Máscara:** Expande la máscara unos píxeles hacia la zona sana. Es **esencial (valor 90-180 según modelo)** para que la IA disponga de margen para mezclar la textura reparada sin dejar un "corte" visible.

### Inferencia y Calibración
* **Mejorar / Proteger Rostros:** Activa el aislamiento facial mediante detección de cabezas. Los rostros sanos se reintegran desde el original con bordes difuminados, evitando que la IA cambie las facciones o la identidad de las personas.
* **Fuerza (Strength):** En 0.99 la IA limpia el daño por completo. Si se baja a 0.85-0.90, permite que se transparente parte de la estructura subyacente.
* **Usar enfoque local (Padding Crop):** En fotos grandes con manchas pequeñas, recorta y procesa la región en alta resolución antes de reinsertarla.

---

## Consejos de Uso
1. **Edición Manual:** Si la detección automática pasa por alto alguna rozadura o mancha fina, usa el pincel blanco sobre el editor para añadirla a la máscara antes de pulsar *Reparar*.
2. **Preservación de Tono Automática:** El sistema aplica automáticamente una normalización en espacio de color LAB para evitar que la IA sobresature la imagen o introduzca tintes cálidos/anaranjados artificiales, respetando la película química de la época.
3. **Historial de Resultados:** La pestaña de historial guarda los últimos 10 resultados de la sesión. Puedes descargar cada imagen individualmente o el ZIP completo desde la sección *Mis Descargas*.
4. **Procesamiento Masivo:** Si tienes carpetas enteras de un mismo álbum, utiliza la pestaña **Procesamiento por Lotes** para dejar el clúster trabajando en segundo plano y recibir un aviso por correo electrónico al finalizar.
5. **Sesiones de Usuario:** Cada usuario tiene su propio espacio seguro en el servidor. Los archivos se guardan bajo su nombre de usuario y no se mezclan con los de otros usuarios.
"""


PROMPT_SDXL = "photo restoration, realistic, same style, clean stains"
NEG_SDXL = "blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text, watermark, signature, logo, text logo, date, extra person, multiple people, group, cloned face, duplicate, extra limbs, extra face, bad anatomy, mutated hands, deformed face, photo caption"
PROMPT_QWEN = (
    "Task: Professional Photo Restoration. "
    "Action: Remove all brown mud stains, dirt spots, and water damage overlays from the image. "
    "Detail: Inpaint the cleaned areas to seamlessly match the original background texture and lighting. "
    "Constraint: Strictly preserve the original facial features, identity, and skin texture of the people. If there is any text in the image restore it to be clear and legible without altering the font style. "
    "Do not smooth the skin excessively. Do not change the clothing style. "
)
NEG_QWEN = (
    "mud, dirt, stains, scratches, blur, noise, "
    "cartoon, illustration, painting, 3d render, anime, "
    "distorted faces, bad anatomy, extra fingers, changed identity, "
    "makeup, plastic skin, smooth skin, ai generated look, "
    "overexposed, underexposed, color drift"
)

standard_font = ["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]

css = """
/* Fuente estándar global */
body, .gradio-container { font-family: system-ui, -apple-system, sans-serif !important; }

.gradio-container { 
    padding-bottom: 60vh !important; 
}

#logo_box { text-align: center; width: 100% !important; } 
#logo_box img { margin: 0 auto; height: 100px; }

/* 1. ELIMINAR TEXTO "BUILT WITH GRADIO" */
footer { display: none !important; }

/* 2. Eliminar fondos de cajas de etiquetas y bloques */

/* 3. Estilo de etiquetas
.block-label span, .block-label {
    background: transparent !important;
    color: #1b4332 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    border: none !important;
}

/* 4. Botones Verde Oscuro */
.primary { background-color: #1b4332 !important; border-color: #1b4332 !important; color: white !important; }
.primary:hover { background-color: #2d6a4f !important; }

/* 5. Color de Sliders y Checkboxes */
input[type='range'] { accent-color: #1b4332 !important; }
input[type='checkbox']:checked { background-color: #1b4332 !important; border-color: #1b4332 !important; }

/* 6. Adaptación del Slider de comparación */
.gradio-container .gr-comparison-slider { height: auto !important; }
.gradio-container img { object-fit: contain !important; }
"""

theme = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate",
    font=standard_font,
).set(
    button_primary_background_fill="#1b4332",
    button_primary_background_fill_hover="#2d6a4f",
    slider_color="#1b4332",
    block_background_fill="transparent",
    block_label_background_fill="transparent",
    block_label_text_color="#1b4332",
    block_border_width="0px",
)

with gr.Blocks(title="Proyecto Rebrot") as demo:
    
    # ESTADOS GLOBALES
    current_user = gr.State("")
    st_boxes = gr.State()
    st_f_mask = gr.State()
    st_r_mask = gr.State()
    history_state = gr.State([])

    # --- VISTA 1: LOGIN ---
    with gr.Column(visible=True) as login_view:
        gr.Image(value="static/logo.png", show_label=False, elem_id="logo_box", interactive=False)
        gr.Markdown("## Acceso al Sistema")
        gr.Markdown("Introduce un nombre de usuario. Esto nos permite guardar tus archivos de forma segura para que puedas descargarlos más tarde.")
        
        #with gr.Row():
        tb_username = gr.Textbox(label="Nombre de Usuario (Sin espacios)", placeholder="Ej: maria_gomez")
        btn_login = gr.Button("Entrar", variant="primary")
        lbl_login_err = gr.Markdown()

    # --- VISTA 2: APLICACIÓN PRINCIPAL ---
    with gr.Column(visible=False) as main_view:
        gr.Image(value="static/logo.png", show_label=False, elem_id="logo_box", interactive=False)   
        lbl_welcome = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=3):
                lbl_welcome = gr.Markdown()
            with gr.Column(scale=2):
                # VISOR DE COLA
                queue_info = gr.Markdown("🟢 **Servidor Listo** | Imágenes en cola: 0")
            with gr.Column(scale=1):
                est_time_info = gr.Markdown("⏳ T. espera: 0s")
        
        with gr.Tabs():
            # --- PESTAÑA 1: MANUAL ---
            with gr.Tab("🛠️ Edición Manual"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        im_input = gr.Image(label="1. Subir Original", type="filepath", sources=["upload", "clipboard"])
                    with gr.Column(scale=1):
                        im_yolo = gr.Image(label="2. Vista Previa YOLO (Zonas)", interactive=False)
                    with gr.Column(scale=1):
                        im_mask_editor = gr.ImageEditor(
                            label="3. Máscara Final (Editable)", type="pil", layers=False,
                            brush=gr.Brush(colors=["#FFFFFF"], default_size=20, color_mode="fixed"),
                            eraser=gr.Eraser(default_size=20), sources=[]
                        )
                with gr.Accordion("Parámetros de Detección de Daños", open=False):
                    with gr.Row():
                        sl_conf = gr.Slider(0, 1, 0.1, step=0.01, label="Confianza Detección (Sensibilidad)")
                        sl_dil = gr.Slider(0, 500, 200, step=10, label="Dilatación Máscara (Grosor del parche)")
                        dd_seg_manual = gr.Dropdown(
                            choices=[("YOLO + SAM", "yolo_sam"), ("U-Net", "unet"), ("Segformer", "segformer")], 
                            value="yolo_sam", 
                            label="Modelo de Segmentación"
                        )
                        dd_seg_manual.change(
                            fn=update_manual_dilation,
                            inputs=[dd_seg_manual],
                            outputs=[sl_dil]
                        )

                btn_det = gr.Button("Detectar Daños", variant="primary")
                lbl_stat = gr.Markdown("Sube una imagen y pulsa Detectar para comenzar.")
                
                with gr.Accordion("Configuración de Inferencia", open=False):
                    with gr.Row():
                            t_prompt = gr.Textbox(label="Prompt", value=PROMPT_SDXL, lines=3)
                            t_neg = gr.Textbox(label="Negative Prompt", value=NEG_SDXL, lines=3)
                    with gr.Row():
                        sl_str = gr.Slider(0, 1, 0.99, step=0.01, label="Fuerza de cambio (Strength)")
                        sl_guid = gr.Slider(0, 50, 7.0, step=0.5, label="Fidelidad al prompt (Guidance)")
                        sl_steps = gr.Slider(1, 100, 20, step=1, label="Pasos de refinado (Steps)")
                    with gr.Row():
                        sl_pad_sz = gr.Slider(0, 100, 32, label="Tamaño del margen (Padding Size)")
                        chk_pad = gr.Checkbox(label="Usar enfoque local (Use Padding)", value=False)
                        chk_enh = gr.Checkbox(label="Mejorar Rostros", value=True)
                    dd_model = gr.Dropdown(choices=["sdxl", "flux", "qwen"], value="sdxl", label="Modelo de IA")
                    dd_model.change(
                        fn=update_prompts_by_model,
                        inputs=[dd_model],
                        outputs=[t_prompt, t_neg, sl_steps]
                    )
                
                btn_rest = gr.Button("Reparar", variant="primary")
                
                gr.Markdown("### Comparador Interactivo")
                im_comparison = gr.ImageSlider(label="Desliza para comparar (Antes vs Después)", type="filepath", interactive=False, buttons=["download", "fullscreen"])
                with gr.Row():
                    im_side_before = gr.Image(label="Original", interactive=False)
                    im_side_after = gr.Image(label="Reparada", interactive=False)
                
                gr.Markdown("### Historial de la sesión manual")
                gallery_history = gr.Gallery(label="Últimos resultados", columns=5, rows=2, height="auto", allow_preview=True, object_fit="contain")

                im_input.change(fn=reset_state, inputs=[], outputs=[im_yolo, im_mask_editor, lbl_stat, st_boxes, st_f_mask, st_r_mask, im_comparison, im_side_before, im_side_after])
                im_input.clear(fn=reset_state, inputs=[], outputs=[im_yolo, im_mask_editor, lbl_stat, st_boxes, st_f_mask, st_r_mask, im_comparison, im_side_before, im_side_after])
                btn_det.click(fn=ui_detect, inputs=[im_input, sl_conf, sl_dil, dd_seg_manual], outputs=[im_yolo, im_mask_editor, lbl_stat, st_boxes, st_f_mask, st_r_mask])
                btn_rest.click(
                    fn=ui_restore, 
                    inputs=[im_input, im_mask_editor, t_prompt, t_neg, sl_str, sl_guid, sl_steps, chk_pad, sl_pad_sz, chk_enh, dd_model, st_boxes, st_f_mask, st_r_mask, history_state], 
                    outputs=[im_comparison, im_side_before, im_side_after, gallery_history, history_state]
                )
                
            # --- PESTAÑA 2: LOTES ---
            with gr.Tab("📦 Procesamiento por Lotes"):
                files_in = gr.File(file_count="multiple", label="Subir Imágenes para Lote", type="filepath")
                with gr.Accordion("Configuración Específica por Modelo", open=False):
                    with gr.Tabs():
                        with gr.Tab("Parámetros SDXL"):
                            with gr.Row():
                                t_prompt_sdxl = gr.Textbox(label="Prompt SDXL", value=PROMPT_SDXL, lines=2)
                                t_neg_sdxl = gr.Textbox(label="Negative Prompt SDXL", value=NEG_SDXL, lines=3)
                            with gr.Row():
                                sl_str_sdxl = gr.Slider(0, 1, 0.99, label="Fuerza (Strength)")
                                sl_guid_sdxl = gr.Slider(0, 50, 7.0, label="Fidelidad al prompt (Guidance)")
                                sl_steps_sdxl = gr.Slider(1, 100, 20, label="Pasos (Steps)")

                        with gr.Tab("Parámetros FLUX"):
                            with gr.Row():
                                t_prompt_flux = gr.Textbox(label="Prompt Flux", value=PROMPT_SDXL, lines=2)
                                t_neg_flux = gr.Textbox(label="Negative Prompt Flux", value=NEG_SDXL, lines=3)
                            with gr.Row():
                                sl_str_flux = gr.Slider(0, 1, 0.99, label="Fuerza (Strength)")
                                sl_guid_flux = gr.Slider(0, 50, 3.5, label="Fidelidad al prompt (Guidance)")
                                sl_steps_flux = gr.Slider(1, 100, 20, label="Pasos (Steps)")

                        with gr.Tab("Parámetros QWEN"):
                            with gr.Row():
                                t_prompt_qwen = gr.Textbox(label="Prompt Qwen", value=PROMPT_QWEN, lines=4)
                            with gr.Row():
                                t_neg_qwen = gr.Textbox(label="Negative Prompt Qwen", value=NEG_QWEN, lines=2)
                            with gr.Row():
                                sl_steps_qwen = gr.Slider(1, 100, 8, label="Pasos (Steps)")
                    
                with gr.Accordion("Configuración Común de Lote", open=False):
                    with gr.Row():
                        sl_conf_b = gr.Slider(0, 1, 0.1, label="Confianza Detección")
                        sl_dil_b = gr.Slider(0, 500, 200, label="Dilatación Máscara")
                        chk_default_dil_b = gr.Checkbox(label="Usar dilatación por defecto", value=True)
                        chk_enh_b = gr.Checkbox(label="Mejorar Rostros", value=True)
                    with gr.Row():
                        dd_model_b = gr.Dropdown(choices=["sdxl", "flux", "qwen"], value=["sdxl"], multiselect=True, label="Modelos de reparación")
                        dd_seg_batch = gr.Dropdown(
                            choices=[("YOLO + SAM", "yolo_sam"), ("U-Net", "unet"), ("Segformer", "segformer")], 
                            value=["yolo_sam"], 
                            multiselect=True, 
                            label="Modelos de detección de daños"
                        )
                        chk_default_dil_b.change(
                            fn=toggle_batch_dilation,
                            inputs=[chk_default_dil_b],
                            outputs=[sl_dil_b]
                        )
                with gr.Row():
                    tb_email_batch = gr.Textbox(
                        label="Notificar por correo al terminar", 
                        placeholder="tu@email.com (Opcional)",
                    )
                    tb_batch_name = gr.Textbox(
                        label="Nombre del lote", 
                        #value="lote_rebrot", 
                        placeholder="Escribe un nombre para identificar este lote (Opcional)",
                    )
                gr.Markdown("Se enviará un aviso cuando el lote esté listo para descargar.")

                btn_batch = gr.Button("Iniciar Procesamiento", variant="primary")
                batch_log = gr.Markdown("Esperando inicio...") 
                file_out = gr.File(label="Descarga Directa (Aparecerá al terminar)")
                
                btn_batch.click(
                    fn=batch_process, 
                    inputs=[
                        files_in, sl_conf_b, sl_dil_b, chk_enh_b, chk_default_dil_b, dd_model_b, dd_seg_batch,
                        t_prompt_sdxl, t_neg_sdxl, sl_str_sdxl, sl_guid_sdxl, sl_steps_sdxl,
                        t_prompt_flux, t_neg_flux, sl_str_flux, sl_guid_flux, sl_steps_flux,
                        t_prompt_qwen, t_neg_qwen, sl_steps_qwen,
                        current_user, tb_email_batch, tb_batch_name
                    ], 
                    outputs=[file_out, batch_log]
                )

            # --- PESTAÑA 3: MIS DESCARGAS ---
            with gr.Tab("📂 Mis Descargas (Historial)"):
                gr.Markdown("### 🗄️ Explorador de Procesamientos")
                
                dd_folder_select = gr.Dropdown(
                    label="Selecciona un procesamiento anterior",
                    choices=[],
                    interactive=True
                )
                btn_refresh_folders = gr.Button("🔄 Actualizar Lista", variant="secondary")
                file_download_zip = gr.File(label="Descargar este lote completo (ZIP)", interactive=False)
                
                gallery_folder_view = gr.Gallery(
                    label="Imágenes reparadas en este lote",
                    columns=4,
                    rows=None,
                    height="auto",
                    object_fit="contain",
                    allow_preview=True
                )

                # --- Lógica de eventos para la pestaña ---
                
                # 1. Función para refrescar el dropdown
                def ui_refresh_folder_list(username):
                    folders = get_user_folders(username)
                    return gr.update(choices=folders, value=folders[0] if folders else None)

                btn_refresh_folders.click(
                    fn=ui_refresh_folder_list,
                    inputs=[current_user],
                    outputs=[dd_folder_select]
                )

                # 2. Función para cargar imágenes cuando se selecciona una carpeta
                def ui_display_folder_content(username, folder_name):
                    images, zip_file = load_images_from_folder(username, folder_name)
                    if not images:
                        return [], None
                    return images, zip_file

                dd_folder_select.change(
                    fn=ui_display_folder_content,
                    inputs=[current_user, dd_folder_select],
                    outputs=[gallery_folder_view, file_download_zip]
                )

            # --- PESTAÑA 4: AYUDA ---
            with gr.Tab("📖 Guía y Ayuda"):
                gr.Markdown(HELP_CONTENT)
                
    timer = gr.Timer(3)
    timer.tick(fn=update_queue_display, outputs=[queue_info, est_time_info])

    # Conexión del botón de Login
    btn_login.click(
        fn=login,
        inputs=[tb_username],
        outputs=[login_view, main_view, current_user, lbl_login_err, lbl_welcome]
    ).then(
        fn=ui_refresh_folder_list,
        inputs=[current_user],
        outputs=[dd_folder_select]
    )

if __name__ == "__main__":
    demo.launch(
        #server_name="127.0.0.1",
        server_name="0.0.0.0",
        server_port=PORT_GUI,
        show_error=True,
        debug=True,
        css=css,
        theme=theme,
        favicon_path="static/logo-cropped.png",
        share=True,
    )