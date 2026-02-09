import os
import glob
import shutil
import time
import threading
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import zipfile
import tempfile
from datetime import datetime
from PIL import Image, ImageOps
from dotenv import load_dotenv
from libs.sam2.model import SAM2
from libs.langsam.model import LangSAMFaceExtractor
from libs.unet.model import UNetInference
from libs.stable_diffusion.impaint.model import SDImpainting
from libs.yolov8.model import YOLOV8
from libs.segformer.model import SegformerInference
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    crop_image,
    delete_files
)

load_dotenv()

# Configuración del dispositivo para modelos
DEVICE = "cuda:0"#os.environ.get("CUDA_DEVICE")
print(f"DEVICE {DEVICE}")

# Cargar modelos
sam_segmentation_model = SAM2(DEVICE)
face_detector = LangSAMFaceExtractor(device=DEVICE)
unet_segmentation_model = UNetInference(DEVICE)
impainting_model = SDImpainting(DEVICE)
yolo_model = YOLOV8(device=DEVICE)
segformer_model = SegformerInference("cuda:1")

print("YOLO models available")
yolo_path = "./tools/trainer/yolov8/runs/detect/full_dataset_yolov8x10_17/weights/best.pt"
print(f"Yolo model to use {yolo_path}")
yolo_model.set_model(yolo_path)


# ====== Estado Global Compartido y Mecanismo de Bloqueo ======
shared_processing_data = []
processing_lock = threading.Lock()
is_processing = False
# ===========================================================

def get_current_processing_state():
    global shared_processing_data, is_processing
    message = "Sube archivos para iniciar el proceso."
    if is_processing:
        message = "⏳ Ya hay un proceso de restauración en curso. Verificando estado..."
    elif shared_processing_data:
        message = "🎉 Último proceso completado."
    return shared_processing_data, message

# Función para comprimir resultados
def zip_results(source_dir, output_filename):
    shutil.make_archive(output_filename, 'zip', source_dir)
    return f"{output_filename}.zip"

# Función principal modificada para manejar archivos subidos y salida ZIP
def handle_processing_click(archivos_subidos, segmentation_models):
    global shared_processing_data, is_processing

    if is_processing:
        return shared_processing_data, "⏳ Ya hay un proceso en curso.", None

    with processing_lock:
        if is_processing:
            return shared_processing_data, "⏳ Ya hay un proceso en curso.", None

        is_processing = True
        shared_processing_data = []

        if not archivos_subidos:
            is_processing = False
            return [], "⚠️ No has subido archivos.", None

        # Crear directorio temporal para guardar los resultados de este lote
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(tempfile.gettempdir(), f"restauracion_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Preparar datos iniciales
        rutas_archivos = [f.name for f in archivos_subidos]
        
        for ruta in rutas_archivos:
            for seg_model in segmentation_models:
                shared_processing_data.append([os.path.basename(ruta), seg_model, "✨ Pendiente", ""])

        yield shared_processing_data, "✅ Iniciando procesamiento...", None

        # Procesar cada archivo
        for fila_idx, (nombre_archivo, modelo, _, _) in enumerate(shared_processing_data):
            # Encontrar la ruta completa original basada en el nombre
            ruta_original = next((r for r in rutas_archivos if os.path.basename(r) == nombre_archivo), None)
            
            if not ruta_original: continue

            print(f"Inicia proceso para {nombre_archivo} con {modelo}")
            shared_processing_data[fila_idx][2] = "⏳ Procesando..."
            yield shared_processing_data, f"⏳ Procesando {nombre_archivo}...", None
            
            begin = time.time()

            try:
                # Definir rutas de salida dentro de la carpeta temporal
                nombre_base = os.path.splitext(nombre_archivo)[0]
                
                # Cargar imagen
                image = cv2.imread(ruta_original)
                if image is None: raise Exception("No se pudo leer la imagen")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # --- Lógica de Detección de Rostros ---
                full_face_mask, face_boxes = face_detector(ruta_original, return_results="both", mask_multiplier=255)
                full_face_mask = fill_little_spaces(full_face_mask, 120)
                
                # Guardar máscara rostros
                dilated_full_face_mask_img = Image.fromarray(soften_contours(full_face_mask, 0)).convert("L")
                path_face_mask = os.path.join(output_dir, f"{nombre_base}_face_mask_{modelo}.png")
                dilated_full_face_mask_img.save(path_face_mask)
                
                full_face_mask_pil = Image.fromarray(full_face_mask).convert("L")
                full_face_mask_pil = ImageOps.autocontrast(full_face_mask_pil)

                # --- Lógica de Segmentación ---
                binary_mask = None
                kernel_size_contours = 0

                if modelo == "YOLO+SAM":
                    kernel_size_contours = 100
                    _, boxes = yolo_model.get_bounding_box(0.1, ruta_original)
                    
                    if len(boxes) <= 0: # Si no hay daños, guardar original
                        Image.fromarray(image).save(os.path.join(output_dir, f"{nombre_base}_RESTORED_{modelo}.png"))
                        shared_processing_data[fila_idx][2] = "✅ Sin daños detectados"
                        shared_processing_data[fila_idx][3] = f"{round(time.time()-begin, 3)}"
                        continue

                    masks = sam_segmentation_model.get_mask_by_bounding_boxes(boxes=boxes, image=image)
                    numpy_masks = [mask.cpu().numpy() for mask in masks]
                    combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
                    for m in numpy_masks: combined_mask = np.logical_or(combined_mask, m)
                    binary_mask = generate_binary_mask(combined_mask)

                elif modelo == "UNet":
                    kernel_size_contours = 50
                    binary_mask = unet_segmentation_model.get_mask(image=image)
                
                elif modelo == "SegFormer":
                    kernel_size_contours = 90
                    array = np.array(Image.open(ruta_original))
                    binary_mask = segformer_model.get_mask(array)

                # Guardar máscara original
                Image.fromarray(binary_mask).save(os.path.join(output_dir, f"{nombre_base}_MASK_ORIGINAL_{modelo}.png"))

                # Refinamiento de máscara
                refined = delete_irrelevant_detected_pixels(binary_mask)
                refined = fill_little_spaces(refined)
                dilated = soften_contours(refined, kernel_size_contours)
                
                general_mask = Image.fromarray(refined, mode='L')
                
                # Restar rostros de la máscara de daño
                mask1_np = np.array(general_mask)
                if dilated_full_face_mask_img.size != general_mask.size:
                    mask1_np = np.array(general_mask.resize(dilated_full_face_mask_img.size, Image.NEAREST))
                
                mask2_np = np.array(dilated_full_face_mask_img)
                result_bool = (mask1_np > 0) & ~(mask2_np > 0)
                result_np = np.uint8(result_bool) * 255
                
                # Guardar máscara sin rostros
                Image.fromarray(result_np, mode='L').save(os.path.join(output_dir, f"{nombre_base}_ERASED_FACE_MASK_{modelo}.png"))

                # Dilatación final para inpainting
                dilated_final = soften_contours(result_np, kernel_size_contours)
                dilated_final = soften_contours(dilated_final, kernel_size_contours) # Doble pasada como en original
                
                path_refined_mask = os.path.join(output_dir, f"{nombre_base}_MASK_REFINED_{modelo}.png")
                Image.fromarray(dilated_final, mode='L').save(path_refined_mask)

                # --- Inpainting ---
                print(f"Inpainting {nombre_archivo}...")
                new_image = impainting_model.impaint(
                    image_path=ruta_original,
                    mask_path=path_refined_mask,
                    prompt="photo restoration, realistic, same style, clean stains",
                    strength=0.99,
                    guidance=7,
                    padding_mask_crop=None,
                    steps=20,
                    negative_prompt="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text, watermark, signature, logo, text logo, date, extra person, multiple people, group, cloned face, duplicate, extra limbs, extra face, bad anatomy, mutated hands, deformed face, photo caption"
                )

                # Componer Rostros Originales
                original_pil = Image.fromarray(image).convert("RGBA")
                result_crop = new_image.convert("RGBA")
                new_image = Image.composite(original_pil, result_crop, full_face_mask_pil)

                # Mejorar rostros (Enhance)
                # Nota: Pasamos output_dir como folder para guardar los crops temporales allí
                new_image = __enhance_faces(image, Image.fromarray(binary_mask), face_boxes, new_image, output_dir + os.sep)

                # Guardar resultado final
                new_image.save(os.path.join(output_dir, f"{nombre_base}_RESTORED_{modelo}.png"))

                end = time.time()
                shared_processing_data[fila_idx][2] = "✅ Restaurado"
                shared_processing_data[fila_idx][3] = f"{round(end-begin, 3)}"

            except Exception as e:
                print(f"Error en {nombre_archivo}: {e}")
                shared_processing_data[fila_idx][2] = f"❌ Error: {e}"

            yield shared_processing_data, f"✅ Procesado {nombre_archivo}...", None

        # Empaquetar resultados
        zip_path = zip_results(output_dir, os.path.join(tempfile.gettempdir(), f"resultados_{timestamp}"))
        is_processing = False
        
        yield shared_processing_data, "🎉 Proceso completado. Descarga el ZIP.", zip_path


def __enhance_faces(original_image, binary_mask, face_boxes, inpainted_image, folder):
    original_binary_mask = np.array(binary_mask)
    height, width = original_image.shape[:2]
    padding = 10
    
    for i in range(len(face_boxes)):
        # (Tu lógica original de enhance faces se mantiene igual, 
        # solo asegúrate de que 'folder' termine en separador / o \)
        xmax, xmin, ymax, ymin = map(int, face_boxes[i])
        x1, y1 = max(0, xmin - padding), max(0, ymin - padding)
        x2, y2 = min(width, xmax + padding), min(height, ymax + padding)

        face = crop_image(original_image, x1, y1, x2, y2)
        face_mask = crop_image(original_binary_mask, x1, y1, x2, y2)

        if np.any(face_mask == 255) and np.sum(face_mask == 255) / face_mask.size < 0.4:
            face_img = Image.fromarray(face)
            face_path = folder + f"temp_face_{i}.png"
            face_img.save(face_path)
            
            mask_img = Image.fromarray(face_mask)
            mask_path = folder + f"temp_face_mask_{i}.png"
            mask_img.save(mask_path)

            enhanced_face = impainting_model.impaint(
                image_path=face_path,
                mask_path=mask_path,
                prompt="clear skin, smooth face, realistic, natural lighting, photo restoration, high detail, close-up portrait",
                strength=0.99,
                guidance=7,
                steps=20,
                negative_prompt="blemishes, scars, acne, skin spots, dirty texture, unnatural lighting, distortions, blurry, extra eyes, deformed face, plastic look, overexposed",
                padding_mask_crop=None
            )

            enhanced_face_mask = face_detector(face_path, return_results="mask", mask_multiplier=255)
            enhanced_face_mask = fill_little_spaces(enhanced_face_mask, 65)

            enhanced_face = enhanced_face.convert("RGBA")
            mask_array = np.array(enhanced_face_mask)
            rgba_array = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
            rgba_array[..., :3] = np.array(enhanced_face)[..., :3]
            rgba_array[..., 3] = mask_array

            enhanced_face_with_transparency = Image.fromarray(rgba_array, mode="RGBA")
            inpainted_image.paste(enhanced_face_with_transparency, (x2, y1-enhanced_face.size[1]), enhanced_face_with_transparency)

    return inpainted_image

def export_csv():
    if not shared_processing_data: return None
    df = pd.DataFrame(shared_processing_data, columns=["Archivo", "Modelo", "Estado", "Tiempo"])
    output_path = "estado_procesamiento.csv"
    df.to_csv(output_path, index=False)
    return output_path

# --- INTERFAZ GRADIO ---
with gr.Blocks(title="AI-Impainter: Restauración por Lotes") as demo:
    gr.Markdown("# 🎨 Restauración de Fotos por Lotes")
    gr.Markdown("Sube tus imágenes, procésalas en el servidor y descarga los resultados.")

    # 1. Componente de Subida (Sustituye al FileExplorer)
    upload_files = gr.File(
        file_count="multiple",
        file_types=["image"],
        label="📂 1. Arrastra aquí tus fotos o haz clic para subir",
        type="filepath" # Gradio maneja la subida temporal
    )

    status_message = gr.Textbox(label="Estado", interactive=False)
    segmentation_models = gr.CheckboxGroup(["YOLO+SAM", "UNet", "SegFormer"], value=["YOLO+SAM"], label="Modelos")
    
    # 2. Botón de Proceso
    procesar_button = gr.Button("✨ 2. Iniciar Restauración ✨", variant="primary")

    # 3. Salida de Datos
    output_tabla = gr.Dataframe(label="📊 Progreso", headers=["Archivo", "Modelo", "Estado", "Tiempo"], interactive=False)
    
    # 4. Componente de Descarga (ZIP Final)
    zip_output = gr.File(label="📥 3. Descargar Resultados (ZIP)")

    # Eventos
    demo.load(fn=get_current_processing_state, outputs=[output_tabla, status_message], queue=False)
    
    procesar_button.click(
        fn=handle_processing_click,
        inputs=[upload_files, segmentation_models],
        outputs=[output_tabla, status_message, zip_output]
    )

    exportar_csv_button = gr.Button("📥 Descargar Log CSV")
    csv_file_output = gr.File(label="CSV Log")
    exportar_csv_button.click(fn=export_csv, inputs=[], outputs=csv_file_output)

demo.launch(
    server_name="0.0.0.0", 
    server_port=7860, 
    debug=True, 
#    auth=(os.environ.get("APP_USER"), os.environ.get("APP_PASSWORD"))
)