import gradio as gr
import os
import tempfile
import shutil
from PIL import Image
from api_client import detect_damage, restore_image

def save_temp_image(image_pil, suffix=".png"):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image_pil.save(t.name)
    return t.name

# --- LÓGICA MANUAL ---
def ui_detect(img_path, conf, dilat):
    if not img_path: 
        return None, None, "Falta imagen", None, None, None
    try:
        yolo_prev, mask, boxes, f_mask, r_mask = detect_damage(img_path, conf, dilat)
        
        if mask is None: 
            return None, None, "No se detectó daño", None, None, None
        
        original_img = Image.open(img_path).convert("RGB")
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


def ui_restore(img_path, editor_data, prompt, neg, strn, guid, steps, pad, pad_sz, enh, cnet, model_var, state_boxes, state_f_mask, state_r_mask):
    if not img_path or not editor_data: 
        return None, None
    
    # Extraemos la máscara del editor (incluso con layers=False, Gradio usa la lista 'layers')
    if editor_data.get('layers') and len(editor_data['layers']) > 0:
        # Obtenemos la capa de dibujo y la convertimos a blanco y negro (L)
        mask_pil = editor_data["layers"][0].convert("L")
        # Aseguramos que lo que es traslúcido sea blanco puro para el inpainting
        mask_pil = mask_pil.point(lambda p: 255 if p > 0 else 0)
    else:
        bg = editor_data["background"]
        mask_pil = Image.new("L", bg.size, 0)
    
    params = {
        "prompt": prompt, "negative_prompt": neg, "strength": strn,
        "guidance": guid, "steps": steps, "use_padding": pad,
        "padding_crop": pad_sz, "enhance_faces": enh,
        "use_controlnet": cnet, "model_variant": model_var,
    }
    
    try:
        res_img = restore_image(img_path, mask_pil, state_f_mask, state_r_mask, state_boxes, params)
        out_path = save_temp_image(res_img)
        return img_path, out_path
    
    except Exception as e:
        raise gr.Error(f"Error Inpainting: {e}")
    

# --- LÓGICA BATCH ---
def batch_process(files, conf, strn, steps, progress=gr.Progress()):
    if not files: return None
    
    base_temp_dir = tempfile.mkdtemp()
    images_dir = os.path.join(base_temp_dir, "imagenes_restauradas")
    os.makedirs(images_dir, exist_ok=True)
    
    for f in progress.tqdm(files, desc="Procesando lote"):
        try:
            file_path = f if isinstance(f, str) else f.name
            # 1. Detectar
            _, mask, boxes, f_mask, r_mask = detect_damage(f.name, conf, 10)

            if mask is None: # No hay daño
                fname = os.path.basename(file_path)
                shutil.copy(file_path, os.path.join(images_dir, f"no_damage_{fname}"))
                continue 
            
            # 2. Restaurar
            params = {
                "prompt": "photo restoration, realistic, same style, clean stains",
                "negative_prompt": "blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text, watermark, signature, logo, text logo, date, extra person, multiple people, group, cloned face, duplicate, extra limbs, extra face, bad anatomy, mutated hands, deformed face, photo caption",
                "strength": strn, 
                "guidance": 7.0, 
                "steps": steps,
                "use_padding": False, 
                "padding_crop": 32, 
                "enhance_faces": True
            }
            res_img = restore_image(f.name, mask, f_mask, r_mask, boxes, params)
            if res_img.mode in ("RGBA", "P"):
                res_img = res_img.convert("RGB")

            # Guardar
            fname = os.path.basename(file_path)
            res_img.save(os.path.join(images_dir, f"restored_{fname}"))

        except Exception as e:
            print(f"Error en {f.name}: {e}")
            
    # ZIP
    zip_filename_base = os.path.join(base_temp_dir, "resultados_rebrot")
    full_zip_path = shutil.make_archive(zip_filename_base, 'zip', images_dir)
    return full_zip_path

# --- INTERFAZ ---
css = """
#logo_box {
    text-align: center;
    width: 100% !important;
} 
logo_box img {
    margin: 0 auto;
    height: 100px
}"""

with gr.Blocks(css=css, title="Rebrot Project") as demo:
#    gr.Markdown("# Rebrot: Restauración de Imágenes (Cliente/Servidor)")
    gr.Image(value="static/logo.png", show_label=False, elem_id="logo_box")   
    # Estados para pestaña Manual
    st_boxes = gr.State()
    st_f_mask = gr.State()
    st_r_mask = gr.State()
    
    with gr.Tabs():
        # --- PESTAÑA 1: MANUAL ---
        with gr.Tab("🛠️ Edición Manual"):
            with gr.Row():
                sl_conf = gr.Slider(0, 1, 0.1, step=0.01, label="Confianza Detección")
                sl_dil = gr.Slider(0, 500, 200, step=10, label="Dilatación Máscara")
            
            btn_det = gr.Button("🔍 Detectar Daños", variant="primary")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    im_input = gr.Image(label="1. Subir Original", type="filepath")
                
                with gr.Column(scale=1):
                    im_yolo = gr.Image(label="2. Vista Previa YOLO (Zonas)", interactive=False)
                
                with gr.Column(scale=1):
                    im_mask_editor = gr.ImageEditor(
                        label="3. Máscara Final (Editable)", 
                        type="pil", 
                        layers=False,
                        brush=gr.Brush(colors=["#FFFFFF"], default_size=20, color_mode="fixed"),
                        eraser=gr.Eraser(default_size=20),
                        sources=[]
                    )
            
            lbl_stat = gr.Markdown("Sube una imagen y pulsa Detectar para comenzar.")
            
            with gr.Row():
                    t_prompt = gr.Textbox(label="Prompt", value="photo restoration, realistic, same style, clean stains")
                    t_neg = gr.Textbox(label="Negative Prompt", value="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text, watermark, signature, logo, text logo, date, extra person, multiple people, group, cloned face, duplicate, extra limbs, extra face, bad anatomy, mutated hands, deformed face, photo caption")
                
            with gr.Row():
                sl_str = gr.Slider(0, 1, 0.99, step=0.01, label="Strength")
                sl_guid = gr.Slider(0, 50, 7.0, step=0.5, label="Guidance")
                sl_steps = gr.Slider(1, 100, 20, step=1, label="Steps")
            
            with gr.Row():
                chk_pad = gr.Checkbox(label="Use Padding", value=False)
                sl_pad_sz = gr.Slider(0, 100, 32, label="Padding Size")
                chk_enh = gr.Checkbox(label="Mejorar Rostros", value=True)
                chk_cnet = gr.Checkbox(label="Activar ControlNet", value=False)
            
            dd_model = gr.Dropdown(
                    choices=["sdxl", "flux", "qwen"], 
                    value="sdxl", 
                    label="Modelo de IA (Inpainting)",
            )
            btn_rest = gr.Button("🎨 Restaurar", variant="primary")
            
            gr.Markdown("### ⚖️ Comparación de Resultados")
            with gr.Row():
                im_compare_orig = gr.Image(label="Antes", interactive=False)
                im_res = gr.Image(label="Después (Restaurada)", interactive=False)
            
            # 1. Detección
            btn_det.click(
                fn=ui_detect, 
                inputs=[im_input, sl_conf, sl_dil], 
                outputs=[im_yolo, im_mask_editor, lbl_stat, st_boxes, st_f_mask, st_r_mask]
            )
            
            # 2. Restauración
            btn_rest.click(
                fn=ui_restore, 
                inputs=[
                    im_input,
                    im_mask_editor,
                    t_prompt,
                    t_neg,
                    sl_str,
                    sl_guid,
                    sl_steps,
                    chk_pad,
                    sl_pad_sz,
                    chk_enh,
                    chk_cnet,
                    dd_model,
                    st_boxes,
                    st_f_mask,
                    st_r_mask
                ], 
                outputs=[im_compare_orig, im_res]
            )
            
        # --- PESTAÑA 2: LOTES ---
        with gr.Tab("📦 Procesamiento por Lotes"):
            files_in = gr.File(file_count="multiple", label="Imágenes", type="filepath")
            btn_batch = gr.Button("🚀 Procesar Todo")
            file_out = gr.File(label="Descargar ZIP")
            
            btn_batch.click(batch_process, [files_in, sl_conf, sl_str, sl_steps], file_out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)