import os
import glob
import shutil
import time
import threading
import cv2
import gradio as gr
import numpy as np
import pandas as pd
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
import patches

load_dotenv()

# Configuración del dispositivo para modelos
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = os.environ.get("DEVICE")
# DEVICE = "cuda:0"
print(f"DEVICE {DEVICE}")

# Cargar modelos
sam_segmentation_model = SAM2(DEVICE)
face_detector = LangSAMFaceExtractor(device=DEVICE)
unet_segmentation_model = UNetInference(DEVICE)
impainting_model = SDImpainting(DEVICE)
yolo_model = YOLOV8(device=DEVICE)
segformer_model = SegformerInference("cuda:1")

#paths = glob.glob("./tools/trainer/yolov8/runs/detect/*/weights/best.pt")
print("YOLO models available")
#print(paths)
yolo_path = "./tools/trainer/yolov8/runs/detect/full_dataset_yolov8x10_17/weights/best.pt"
print(f"Yolo model to use {yolo_path}")
yolo_model.set_model(yolo_path)


# ====== Estado Global Compartido y Mecanismo de Bloqueo ======
shared_processing_data = []
processing_lock = threading.Lock()
is_processing = False
# ===========================================================

# Obtener la ruta del directorio home del usuario
home_directory = os.path.expanduser('~')


# Función para obtener el estado actual del procesamiento compartido (sin controlar el estado del botón)
def get_current_processing_state():
    """
    Retorna el estado actual de los datos de procesamiento compartidos y un mensaje de estado general.
    """
    global shared_processing_data, is_processing
    message = "Selecciona archivos para iniciar el proceso."
    if is_processing:
        message = "⏳ Ya hay un proceso de restauración en curso. Verificando estado..."
    elif shared_processing_data:  # Si hay datos pero no está procesando, mostramos el último estado completado/con error
        message = "🎉 Último proceso completado."

    # Devolvemos los datos de la tabla y el mensaje de estado
    return shared_processing_data, message


# Función que maneja el clic del botón, inicia el procesamiento y actualiza el estado (sin deshabilitar el botón)
def handle_processing_click(lista_elementos_seleccionados, segmentation_models):
    global shared_processing_data, is_processing
    print(f"Archivos seleccionados: {lista_elementos_seleccionados}")
    # Si ya hay un proceso en curso, notificamos y devolvemos el estado actual
    if is_processing:
        # El estado del botón no se controla aquí directamente
        return shared_processing_data, "⏳ Ya hay un proceso de restauración en curso. Por favor espera a que termine."

    # Si no hay proceso en curso, intentamos adquirir el bloqueo
    with processing_lock:
        # Verificación doble dentro del bloqueo
        if is_processing:
            return shared_processing_data, "⏳ Ya hay un proceso de restauración en curso. Por favor espera a que termine."

        is_processing = True  # Marcamos que el proceso está en curso
        shared_processing_data = []  # Limpiamos datos anteriores

        rutas_archivos = []
        if lista_elementos_seleccionados:
            # Ignoramos el primer elemento asumiendo que es la carpeta
            rutas_archivos = [
                ruta for ruta in lista_elementos_seleccionados if os.path.isfile(ruta)]

        # Si no hay archivos seleccionados, salimos
        if not rutas_archivos:
            is_processing = False  # Resetear bandera
            print("ENTRO AQUI")
            return [], "⚠️ No hay archivos seleccionados para procesar."

        # 1. Preparar datos iniciales de la tabla con estado 'Pendiente'
        for ruta in rutas_archivos:
            shared_processing_data.extend(
                [[ruta, seg_model, "✨ Pendiente", ""] for seg_model in segmentation_models])

        # Generar estado inicial: tabla y mensaje
        yield shared_processing_data, "✅ Proceso de restauración iniciado. Procesando archivos..."

        # 2. Procesar cada archivo
        for fila_idx, (ruta_original, modelo, _, _) in enumerate(shared_processing_data):
            print(
                f"Inicia proceso de restauración para imagen {ruta_original} con modelo {modelo}")
            shared_processing_data[fila_idx][2] = "⏳ Procesando..."
            begin = time.time()
            # Generar actualizaciones: tabla y mensaje (sin controlar el botón)
            yield shared_processing_data, f"⏳ Procesando {os.path.basename(ruta_original)} con {modelo}..."

            try:
                # Cargando imagen original
                image = cv2.imread(ruta_original)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detectando rostros
                print("Deteccion de rostros 🎭")
                full_face_mask, face_boxes = face_detector(
                    ruta_original, return_results="both", mask_multiplier=255)

                print(f"Cantidad de rostros: {len(face_boxes)}")

                full_face_mask = fill_little_spaces(full_face_mask, 120)  # 65
                dilated_full_face_mask = soften_contours(
                    full_face_mask, 0)  # TEMPORAL: Se deja mascara de rostros sin dilatado por ahora
                dilated_full_face_mask = Image.fromarray(
                    dilated_full_face_mask).convert("L")

                # Guardando mascara dilatada
                directorio, nombre_completo = os.path.split(ruta_original)
                ruta_base = os.path.join(directorio, '')
                nombre, extension = os.path.splitext(nombre_completo)
                dilated_full_face_mask.save(
                    ruta_base + f"{nombre}_dilated_full_face_mask_{modelo}.png")

                full_face_mask = Image.fromarray(full_face_mask).convert("L")
                full_face_mask = ImageOps.autocontrast(full_face_mask)

                print("Deteccion de rostros exitosa")

                binary_mask = None

                kernel_size_contours = 0
                if modelo == "YOLO+SAM":
                    kernel_size_contours = 100
                    print("YOLO detection started 🔍")
                    yolo_image, boxes = yolo_model.get_bounding_box(
                        0.1, ruta_original)
                    print(
                        f"YOLO detection has finished succesfully. {len(boxes)} boxes")

                    if len(boxes) <= 1:
                        copied_original = Image.open(ruta_original)
                        copied_original = ImageOps.exif_transpose(
                            copied_original)
                        copied_original.save(os.path.join(
                            directorio, f"{nombre}_RESTORED_{modelo}.png"))
                        print("Restauracion de imagen completa")
                        print("===============================")

                        end = time.time()
                        duration = round(end-begin, 3)
                        shared_processing_data[fila_idx][2] = "✅ Restaurado"
                        shared_processing_data[fila_idx][3] = f"{duration}"
                        continue

                    # Generación de la máscara
                    print(f"SAM detection for {len(boxes)} box started 🔬")
                    masks = sam_segmentation_model.get_mask_by_bounding_boxes(
                        boxes=boxes, image=image)
                    print(f"SAM detection has finished successfully")

                    # Mezclar multiples mascaras de SAM
                    numpy_masks = [mask.cpu().numpy() for mask in masks]
                    combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
                    for m in numpy_masks:
                        combined_mask = np.logical_or(combined_mask, m)

                    # Generar mascara binaria
                    binary_mask = generate_binary_mask(combined_mask)
                elif modelo == "UNet":
                    kernel_size_contours = 50
                    print("UNet segmentation started 🔬")
                    binary_mask = unet_segmentation_model.get_mask(image=image)
                    print(f"UNet detection has finished successfully")
                elif modelo == "SegFormer":
                    kernel_size_contours = 90
                    print("SegFormer segmentation started 🔬")
                    array = np.array(Image.open(ruta_original))
                    print(array.shape)
                    # array = np.expand_dims(array, axis = 0)
                    print(array.shape)
                    binary_mask = segformer_model.get_mask(array)
                    print(f"SegFormer detection has finished successfully")

                # Guardar mascara original
                binary_mask_image = Image.fromarray(binary_mask)
                binary_mask_image.save(
                    ruta_base + f"{nombre}_MASK_ORIGINAL_{modelo}.png")

                print("Refining generated mask with OpenCV 🖌")
                refined_binary_mask = delete_irrelevant_detected_pixels(
                    binary_mask)

                without_irrelevant_pixels_mask = fill_little_spaces(
                    refined_binary_mask)

                general_mask = Image.fromarray(
                    without_irrelevant_pixels_mask, mode='L')
                dilated_mask = soften_contours(
                    without_irrelevant_pixels_mask, kernel_size_contours)
                blurred_mask = dilated_mask
                processed_mask = Image.fromarray(blurred_mask, mode='L')
                print("Mask was refined successfully!")

                # Convertir a arrays NumPy
                mask1_np = np.array(general_mask)
                mask2_np = np.array(dilated_full_face_mask)

                if dilated_full_face_mask.size != general_mask.size:
                    processed_mask_resized = general_mask.resize(
                        dilated_full_face_mask.size, Image.NEAREST)
                    mask1_np = np.array(processed_mask_resized)

                # Convertir a booleanos: blancos son mayores a 0 debido a casos de pixel 254
                mask1_bool = mask1_np > 0
                mask2_bool = mask2_np > 0

                # Eliminar píxeles de mask1 donde mask2 es blanco
                result_bool = mask1_bool & ~mask2_bool

                # Convertir el resultado a imagen binaria (0 o 255)
                result_np = np.uint8(result_bool) * 255
                erased_face_mask = Image.fromarray(result_np, mode='L')

                erased_face_mask.save(ruta_base +
                                      f"{nombre}_ERASED_FACE_MASK_{modelo}.png")

                # Dilatamos mascara final
                dilated_mask = soften_contours(
                    result_np, kernel_size_contours)
                dilated_mask = soften_contours(
                    result_np, kernel_size_contours)
                # Guardar máscara refinada
                processed_mask = Image.fromarray(dilated_mask, mode='L')

                ruta_mascara_final = ruta_base + \
                    f"{nombre}_MASK_REFINED_{modelo}.png"
                processed_mask.save(ruta_mascara_final)

                print("SD XL Impainting started 🎨")
                new_image = impainting_model.impaint(
                    image_path=ruta_original,
                    mask_path=ruta_mascara_final,
                    prompt="photo restoration, realistic, same style, clean stains",
                    strength=0.99,
                    guidance=7,
                    padding_mask_crop=None,
                    steps=20,
                    # negative_prompt=""
                    # negative_prompt="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text, new people, text logo, date",
                    negative_prompt="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text, watermark, signature, logo, text logo, date, extra person, multiple people, group, cloned face, duplicate, extra limbs, extra face, bad anatomy, mutated hands, deformed face, photo caption"

                )
                print("SD XL Impainting process finished")

                # new_image.save(
                # ruta_base + f"{nombre}_general_restoration_{modelo}.png")

                print("Conservacion de rostros...")
                # Asegúrate de que ambas imágenes estén en modo RGBA
                original_image = Image.fromarray(image).convert("RGBA")
                result_crop = new_image.convert("RGBA")

                full_face_mask.save(
                    ruta_base + f"{nombre}_full_face_mask_{modelo}.png")

                # Componer: donde la máscara es blanca, tomar de original; donde es negra, dejar el resultado
                new_image = Image.composite(
                    result_crop, original_image, processed_mask)
                
                new_image = Image.composite(
                    original_image, new_image, full_face_mask)

                print("Conservacion de rostros exitosa")

                new_image = __enhance_faces(
                    image, binary_mask_image, face_boxes, new_image, ruta_base)

                ruta_restauracion = os.path.join(
                    directorio, f"{nombre}_RESTORED_{modelo}.png")
                new_image.save(ruta_restauracion)

                print("Restauracion de imagen completa")
                print("===============================")

                end = time.time()
                duration = round(end-begin, 3)
                shared_processing_data[fila_idx][2] = "✅ Restaurado"
                shared_processing_data[fila_idx][3] = f"{duration}"
            except Exception as e:
                # Actualizar estado en caso de error
                if len(shared_processing_data[fila_idx]) > 1:
                    error_message = f"❌ Error: {e}"
                    shared_processing_data[fila_idx][2] = error_message
                    shared_processing_data[fila_idx][3] = "-"
                    print(error_message)
                    print("========================")

            # Generar actualizaciones: tabla y mensaje (sin controlar el botón)
            yield shared_processing_data, f"✅ Archivo {os.path.basename(ruta_original)} con {modelo}..."

        # Después de que el bucle termine
        is_processing = False  # Proceso terminado

        # Último estado: tabla y mensaje final
        yield shared_processing_data, "🎉 Proceso de restauración completado."


def __enhance_faces(original_image, binary_mask, face_boxes, inpainted_image, folder):
    original_binary_mask = np.array(binary_mask)
    # Obtener dimensiones de la imagen
    height, width = original_image.shape[:2]
    padding = 10
    for i in range(len(face_boxes)):
        xmax = int(face_boxes[i][0])
        xmin = int(face_boxes[i][1])
        ymax = int(face_boxes[i][2])
        ymin = int(face_boxes[i][3])

        x1 = max(0, xmin - padding)
        y1 = max(0, ymin - padding)
        x2 = min(width, xmax + padding)
        y2 = min(height, ymax + padding)

        face = crop_image(original_image, x1, y1, x2, y2)
        face_mask = crop_image(
            original_binary_mask, x1, y1, x2, y2)

        # Verifica si manchas en rostros son menores al 40% (si son mas, ya no se da libertad al modelo de hacer la restauracion para evitar halucinacion)
        if np.any(face_mask == 255) and np.sum(face_mask == 255) / face_mask.size < 0.4:
            face = Image.fromarray(face)
            face_image_path = folder + f"face_{i}.png"
            face.save(face_image_path)
            face_mask = Image.fromarray(face_mask)
            face_mask_path = folder + f"face_mask_{i}.png"
            face_mask.save(face_mask_path)

            print(f"SD XL Enhancing Face {i} 🎨")
            enhanced_face = impainting_model.impaint(
                image_path=face_image_path,
                mask_path=face_mask_path,
                # prompt="photo restoration, realistic, same style, clean stains",
                # prompt="clean face without stains",
                prompt="clear skin, smooth face, realistic, natural lighting, photo restoration, high detail, close-up portrait",
                strength=0.99,
                guidance=7,
                steps=20,
                negative_prompt="blemishes, scars, acne, skin spots, dirty texture, unnatural lighting, distortions, blurry, extra eyes, deformed face, plastic look, overexposed",
                # negative_prompt="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text",
                padding_mask_crop=None
            )
            print("SD XL Impainting face finished")

            enhanced_face_mask = face_detector(
                face_image_path, return_results="mask", mask_multiplier=255)
            enhanced_face_mask = fill_little_spaces(enhanced_face_mask, 65)

            # Convert enhanced_face to RGBA for transparency
            enhanced_face = enhanced_face.convert("RGBA")
            enhanced_face.save(folder + f"enhanced_face_{i}.png")

            # Create transparency mask from enhanced_face_mask
            # Convert mask to array for manipulation
            mask_array = np.array(enhanced_face_mask)
            # Create RGBA array where alpha channel is based on the mask
            rgba_array = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
            # Copy RGB channels from enhanced_face
            rgba_array[..., :3] = np.array(enhanced_face)[..., :3]
            # Set alpha channel based on mask (255 where mask is white, 0 where black)
            rgba_array[..., 3] = mask_array

            # Convert back to PIL Image with transparency
            enhanced_face_with_transparency = Image.fromarray(
                rgba_array, mode="RGBA")
            # enhanced_face_with_transparency.save(folder + f"enhanced_transparent_face{i}.png")
            # enhanced_face.save(folder + f"enhanced_face{i}.png")
            inpainted_image.paste(
                enhanced_face_with_transparency, (x2, y1-enhanced_face.size[1]), enhanced_face_with_transparency)

            # Delete images of face and face mask
            #delete_files([face_image_path, face_mask_path])
    return inpainted_image


def export_csv():
    """
    Exporta los datos compartidos de procesamiento como un archivo CSV descargable.
    """
    # global shared_processing_data
    if not shared_processing_data:
        return None  # No hay datos para exportar

    df = pd.DataFrame(shared_processing_data, columns=[
                      "Ruta del Archivo", "Modelo Segmentación", "Estado", "Tiempo (s)"])
    output_path = "estado_procesamiento.csv"
    df.to_csv(output_path, index=False)
    return output_path


# Creamos la interfaz de Gradio
# Añadimos un título para la pestaña del navegador
with gr.Blocks(title="AI-Impainter: Restauración de Fotos de la DANA") as demo:
    gr.Markdown(
        """
        # 🎨 Salvem les fotos - Restauración de Fotos con IA 📸

        Bienvenido a **AI-Impainter**, tu herramienta especializada en devolver la vida a esas preciadas fotos
        afectadas por el barro y el agua de la DANA. ✨

        Selecciona las imágenes que deseas restaurar utilizando el explorador de archivos y
        presiona el botón para iniciar el proceso de restauración con nuestra inteligencia artificial. 🖌️
        """
    )

    # Componente FileExplorer para seleccionar archivos/carpetas.
    file_explorer = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="Sube o arrastra las fotos aquí"
    )
    # Esto solo para debuggear: verás en la consola qué se selecciona en tiempo real
    #file_explorer.change(fn=lambda x: print(f"Selección actual: {x}"), inputs=[file_explorer])

    # Componente Textbox para mostrar mensajes de estado general del proceso.
    # Se actualizará al cargar la página y al presionar el botón.
    status_message = gr.Textbox(
        label="Estado General del Proceso",
        interactive=False,
    )

    # Escoger modelos de segmentacion
    segmentation_models = gr.CheckboxGroup(
        ["YOLO+SAM", "UNet", "SegFormer"], value=["YOLO+SAM", "UNet", "SegFormer"], label="Modelo de segmentación")

    # Definimos el botón de procesamiento
    procesar_button = gr.Button(
        "✨ Iniciar Restauración con IA ✨")  # Botón con emojis

    # Componente Dataframe para mostrar la lista de rutas de archivos y su estado.
    # Se actualizará al cargar la página y al presionar el botón.
    output_tabla_procesamiento = gr.Dataframe(
        label="📊 Estado del Procesamiento de Archivos",  # Etiqueta con emoji
        headers=["Ruta del Archivo", "Modelo Segmentación", "Estado",
                 "Tiempo (s)"],  # Cabeceras de la tabla
        interactive=False,  # La tabla no es editable por el usuario
    )

    # ====== Manejo del Estado Compartido y Concurrencia ======

    # Carga el estado actual al cargar la página.
    demo.load(
        fn=get_current_processing_state,  # Llama a la función que obtiene el estado global
        outputs=[output_tabla_procesamiento, status_message],
        queue=False  # Es importante que los eventos load no se encolen
    )

    # Conecta el botón al manejador del proceso.
    # Actualiza la tabla y el mensaje de estado (sin controlar el estado del botón).
    procesar_button.click(
        fn=handle_processing_click,  # Llama a la función que maneja el clic y el proceso
        # La entrada es el valor actual del file_explorer y modelos de segmentacion
        inputs=[file_explorer, segmentation_models],
        # Las salidas son la tabla y el mensaje de estado
        outputs=[output_tabla_procesamiento, status_message]
        # Gradio gestionará automáticamente el encolamiento si múltiples usuarios presionan el botón
    )

    # Botón para exportar CSV
    exportar_csv_button = gr.Button("📥 Descargar Tabla como CSV")

    # Componente de archivo para descargar el CSV
    csv_file_output = gr.File(label="Archivo CSV generado")

    # Conectar el botón con la función export_csv
    exportar_csv_button.click(
        fn=export_csv,
        inputs=[],
        outputs=csv_file_output
    )


# Lanzamos la interfaz de Gradio.
demo.launch(server_name="127.0.0.1",
            server_port=7860,
            debug=True,
            #auth=(os.environ.get("APP_USER"), os.environ.get("APP_PASSWORD"))
        )
