import io
import cv2
import os
import re
import glob
import gradio as gr
from dotenv import load_dotenv
from PIL import Image, ImageOps
import numpy as np
from libs.sam2.model import SAM2
from libs.langsam.model import LangSAMFaceExtractor
from libs.stable_diffusion.impaint.model import SDImpainting
# from libs.flux.model import FluxImpainting
from libs.blip.model import BLIP
from libs.yolov8.model import YOLOV8
from utils import (
    generate_binary_mask,
    delete_irrelevant_detected_pixels,
    fill_little_spaces,
    soften_contours,
    delete_files,
    crop_image
)
import transformers
import torch
from transformers.generation.configuration_utils import GenerationConfig
from transformers.configuration_utils import PretrainedConfig

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
# Cargando variables de entorno
load_dotenv()

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

# Carpeta temporal para gradio
os.makedirs(os.environ.get("GRADIO_TEMP_DIR"), exist_ok=True)

# Rutas de archivos generados
RUTA_MASCARA = "processed_mask.png"
RUTA_IMAGEN_FINAL = "final_output.png"

# Configuración del dispositivo para modelos
DEVICE = "cuda:0"#os.environ.get("CUDA_DEVICE")
print(f"DEVICE {DEVICE}")

# Cargar modelos
# captioning_model = BLIP(DEVICE)
segmentation_model = SAM2(DEVICE)
impainting_model = SDImpainting(DEVICE)
# impainting_model = FluxImpainting(DEVICE)
yolo_model = YOLOV8(device=DEVICE)
face_detector = LangSAMFaceExtractor(device=DEVICE)
# Lista Yolos entrenado

face_mask = None
face_boxes = None


def list_best_pt():
    # Buscar todos los archivos best.pt dentro de cualquier carpeta dentro de detect
    paths = glob.glob("./tools/trainer/yolov8/runs/detect/*/weights/best.pt")

    #paths = ["./tools/trainer/yolov8/runs/detect/full_dataset_yolov8x10_17/weights/best.pt"]

    # Extraer un número si lo hay (por ejemplo, para ordenarlo), o simplemente usar el nombre como clave
    def extract_number(path):
        # Intenta buscar un número al final del nombre del directorio contenedor
        match = re.search(r'detect/([^/]+)', path)
        if match:
            name = match.group(1)
            num_match = re.search(r'(\d+)', name)
            return int(num_match.group(1)) if num_match else 0
        return 0

    # Ordenar (puedes cambiar reverse a False si quieres del más viejo al más nuevo)
    paths.sort(key=extract_number, reverse=True)

    #print(f"Modelos YOLO encontrados: {len(paths)}")

    if paths:
        # Cargar el más "nuevo" según el criterio de orden
        yolo_model.set_model(paths[0])

    return paths

# Setea yolo


def upload_yolo_model(path):
    print(f"Se cambia a modelo {path}")
    yolo_model.set_model(path)

# **Función que se ejecuta al cargar una imagen**


def on_image_load(image_path):
    try:
        # print("BLIP captioning started 👀")
        # caption = captioning_model.generate_caption(
        # image_path)  # Generar el caption usando el path
        caption = ""
        # print("BLIP captioning finished")
        # Retornar el caption para que se muestre en el campo de texto y la ruta del archivo original
        return image_path, None
    except Exception as e:
        print(f"Error en la generación del caption: {e}")
        return "Error en la generación del caption"

# Gradio application

with gr.Blocks(css="""
#logo_box { 
    width: 100% !important;   /* el marco ocupa todo el ancho */
    text-align: center;       /* centramos la imagen */
}
#logo_box img {
    height: 100px;            /* alto fijo de la imagen */
    width: auto;              /* conserva proporción */
}
""") as demo:
    gr.Image(
        value="static/logo.png",
        show_label=False,
        elem_id="logo_box"
    )


    with gr.Row():
        img = gr.Image(label="Input Image", type="filepath")
        img_yolo = gr.Image(label="Yolo Image", type="pil")
        processed_img = gr.Image(
            label="Processed Mask", type="filepath", interactive=True)

    with gr.Row(equal_height=True):
        yolo_model_path = gr.Dropdown(
            choices=list_best_pt(), label="Modelos disponibles", scale=4)
        yolo_confidence = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.1,
            step=0.01,
            label="Confianza",
            scale=1,
            interactive=True
        )
        mask_dilatation = gr.Slider(
            minimum=0, maximum=500, value=200, step=1, label="Dilatacion Mask", interactive=True)

    error_message_detection = gr.Markdown()
    with gr.Row():
        detect_button = gr.Button("Detectar Manchas")

    with gr.Row():
        text_input = gr.Textbox(label="Enter prompt", value="photo restoration, realistic, same style, clean stains",
                                placeholder="Write prompt for impainting...")

    with gr.Row():
        strength = gr.Slider(minimum=0.0, maximum=1.0,
                             value=0.99, label="Strength", interactive=True)
        guidance = gr.Slider(minimum=0.0, maximum=50.0,
                             value=7.0, label="Guidance Scale", interactive=True)
        steps = gr.Slider(minimum=0.0, maximum=100.0, value=20.0,
                          step=1.0, label="Steps", interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                use_padding = gr.Checkbox(
                    label="Use Mask Padding", value=False, interactive=True)
            with gr.Column(scale=4):
                mask_padding_crop = gr.Slider(
                    minimum=0.0, maximum=100.0, value=32.0, label="Mask Padding", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                enhance_faces = gr.Checkbox(
                    label="Enhance Faces", value=True, interactive=True)
            with gr.Column(scale=1):
                see_face_masks = gr.Checkbox(
                    label="See Face Mask", value=False, interactive=True)

    with gr.Row():
        negative_prompt = gr.Textbox(
            label="Negative prompt",
            placeholder="Write negative prompt...",
            value="blurry, distorted, unnatural colors, artifacts, harsh edges, unrealistic texture, visible brush strokes, AI look, text")

    error_message_impaint = gr.Markdown()
    with gr.Row():
        send_button = gr.Button("Impaint Image")

    with gr.Row():
        final_image = gr.Image(label="Final Output", type="filepath")

    gr.Markdown("---")
    gr.Markdown("## Resultados")

    with gr.Row():
        original_img = gr.Image(label="Original Image",
                                type="filepath", interactive=False)
        impainted_img = gr.Image(
            label="Impainted Image", type="filepath", interactive=False)

    def generate_mask_with_yolo(image_path: str, confidence, mask_dilatation):
        global face_mask
        global face_boxes
        try:
            print("YOLO detection started 🔍")
            yolo_image, boxes = yolo_model.get_bounding_box(
                confidence, image_path)
            print(
                f"YOLO detection has finished succesfully. {len(boxes)} boxes")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(
                    "No se pudo cargar la imagen. Verifica la ruta.")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            binary_mask = None
            # Generación de la máscara
            if (len(boxes) > 0):

                # Detectando rostros
                print("Deteccion de rostros 🎭")
                face_mask, face_boxes = face_detector(
                    image_path, return_results="both", mask_multiplier=255)

                print(f"Cantidad de rostros detectados: {len(face_boxes)}")

                face_mask = fill_little_spaces(face_mask, 65)
                face_mask = Image.fromarray(face_mask)
                face_mask.save("face_mask.png")

                print("Deteccion de rostros exitosa")

                print(f"SAM detection for {len(boxes)} box started 🔬")
                masks = segmentation_model.get_mask_by_bounding_boxes(
                    boxes=boxes, image=image)
                print(f"SAM detection has finished successfully")

                # Mezclar multiples mascaras de SAM
                numpy_masks = [mask.cpu().numpy() for mask in masks]

                combined_mask = np.zeros_like(numpy_masks[0], dtype=bool)
                for m in numpy_masks:
                    combined_mask = np.logical_or(combined_mask, m)

                # Generar mascara binaria
                binary_mask = generate_binary_mask(combined_mask)
                binary_mask_image = Image.fromarray(binary_mask)
                binary_mask_image.save("original_mask.png")
                print("Refining generated mask with OpenCV 🖌")
                refined_binary_mask = delete_irrelevant_detected_pixels(
                    binary_mask)
                without_irrelevant_pixels_mask = fill_little_spaces(
                    refined_binary_mask)
                dilated_mask = soften_contours(
                    without_irrelevant_pixels_mask, mask_dilatation)
                blurred_mask = dilated_mask
                print("Image was refined successfully!")

                # Guardar máscara procesada
                processed_mask = Image.fromarray(blurred_mask, mode='L')

                # Convertir a arrays NumPy
                mask1_np = np.array(processed_mask)
                mask2_np = np.array(face_mask)

                # Convertir a booleanos: blancos son 255
                mask1_bool = mask1_np == 255
                mask2_bool = mask2_np == 255

                # Eliminar píxeles de mask1 donde mask2 es blanco
                result_bool = mask1_bool & ~mask2_bool

                # Convertir el resultado a imagen binaria (0 o 255)
                result_np = np.uint8(result_bool) * 255

                # Convertir de vuelta a imagen PIL
                result_img = Image.fromarray(result_np, mode='L')

                # Guardar o mostrar el resultado
                result_img.save(RUTA_MASCARA)

                return yolo_image, RUTA_MASCARA, ""
            else:
                return yolo_image, None, ""
        except Exception as e:
            print(f"Error: {e}")
            return None, None, f"<span style='color:red;'>❌ Error: {str(e)}</span>"

    # **Reiniciar la máscara al cambiar de imagen**
    def reset_mask(image_path):
        delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])
        return None, None, None

    # **Procesar la imagen con la máscara y el texto de entrada**
    def process_final_image(original_image_path, mask_path, text, strength, guidance, steps, negative_prompt, use_padding, padding_mask_crop, enhance_faces, see_face_masks):
        try:
            print("SD XL Impainting started 🎨")
            padding_mask_crop = padding_mask_crop if use_padding else None
            new_image = impainting_model.impaint(
                image_path=original_image_path,
                mask_path=mask_path,
                prompt=text,
                strength=strength,
                guidance=guidance,
                steps=steps,
                negative_prompt=negative_prompt,
                padding_mask_crop=padding_mask_crop
            )
            print("SD XL Impainting process finished")

            print("Conservacion de rostros...")
            # Asegúrate de que ambas imágenes estén en modo RGBA
            result_crop = new_image.convert("RGBA")
            original_image = Image.open(original_image_path)
            original_image_rgba = original_image.convert("RGBA")

            # Convertir face_mask a imagen PIL y asegurarse que tenga valores 0-255
            face_mask_img = Image.open("face_mask.png").convert('L')
            face_mask_img = ImageOps.autocontrast(face_mask_img)

            # Componer: donde la máscara es blanca, tomar de original; donde es negra, dejar el resultado
            new_image = Image.composite(
                original_image_rgba, result_crop, face_mask_img)

            print("Conservacion de rostros exitosa")

            if see_face_masks:
                print("Se veran las mascaras (debug only)")
                # Asegúrate de que result_crop esté en modo RGBA
                result_crop = new_image.convert("RGBA")

                # Convertimos face_mask a imagen PIL en modo 'L'
                face_mask_img = Image.open("face_mask.png").convert('L')

                # Invertimos la máscara si es necesario (según cómo sea tu detector)
                # Asegura que la máscara tenga buen rango
                face_mask_img = ImageOps.autocontrast(face_mask_img)

                # Creamos una imagen blanca del mismo tamaño
                white_image = Image.new(
                    "RGBA", result_crop.size, (255, 255, 255, 255))

                # Componemos: donde la máscara es blanca, se usa white_image; en el resto, se usa result_crop
                new_image = Image.composite(
                    white_image, result_crop, face_mask_img)
            elif enhance_faces:
                print("Mejoraremos los rostros")

                original_binary_mask = Image.open("original_mask.png")
                original_binary_mask = np.array(original_binary_mask)

                image = cv2.imread(original_image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                padding = 10
                for i in range(len(face_boxes)):
                    xmax = int(face_boxes[i][0])
                    xmin = int(face_boxes[i][1])
                    ymax = int(face_boxes[i][2])
                    ymin = int(face_boxes[i][3])

                    x1 = xmin - padding
                    y1 = ymin - padding
                    x2 = xmax + padding
                    y2 = ymax + padding

                    face = crop_image(image, x1, y1, x2, y2)

                    face_mask = crop_image(
                        original_binary_mask, x1, y1, x2, y2)

                    if np.any(face_mask == 255):
                        face = Image.fromarray(face)
                        face.save(f"face_{i}.png")
                        face_mask = Image.fromarray(face_mask)
                        face_mask.save(f"face_mask_{i}.png")

                        print(f"SD XL Enhancing Face {i} 🎨")
                        enhanced_face = impainting_model.impaint(
                            image_path=f"face_{i}.png",
                            mask_path=f"face_mask_{i}.png",
                            #prompt=text,
                            prompt="clean face without stains",
                            strength=strength,
                            guidance=guidance,
                            steps=steps,
                            negative_prompt=negative_prompt,
                            padding_mask_crop=None
                        )
                        print("SD XL Impainting process finished")
                        enhanced_face_path = f"enhanced_face{i}.png"
                        enhanced_face.save(enhanced_face_path)

                        enhanced_face_mask = face_detector(
                            enhanced_face_path, return_results="mask", mask_multiplier=255)
                        enhanced_face_mask = fill_little_spaces(
                            enhanced_face_mask, 65)

                        # Convert enhanced_face to RGBA for transparency
                        enhanced_face = enhanced_face.convert("RGBA")

                        # Create transparency mask from enhanced_face_mask
                        # Convert mask to array for manipulation
                        mask_array = np.array(enhanced_face_mask)
                        # Create RGBA array where alpha channel is based on the mask
                        rgba_array = np.zeros(
                            (*mask_array.shape, 4), dtype=np.uint8)
                        # Copy RGB channels from enhanced_face
                        rgba_array[..., :3] = np.array(enhanced_face)[..., :3]
                        # Set alpha channel based on mask (255 where mask is white, 0 where black)
                        rgba_array[..., 3] = mask_array

                        # Convert back to PIL Image with transparency
                        enhanced_face_with_transparency = Image.fromarray(
                            rgba_array, mode="RGBA")
                        # enhanced_face_with_transparency.save(folder + f"enhanced_transparent_face{i}.png")
                        # enhanced_face.save(folder + f"enhanced_face{i}.png")
                        new_image.paste(
                            enhanced_face_with_transparency, (x2, y1-enhanced_face.size[1]), enhanced_face_with_transparency)

            new_image.save(RUTA_IMAGEN_FINAL)
            return RUTA_IMAGEN_FINAL, RUTA_IMAGEN_FINAL, ""
        except Exception as e:
            print(f"Error: {e}")
            return None, None, f"<span style='color:red;'>❌ Error: {str(e)}</span>"

    def on_clear_processed_mask():
        delete_files([RUTA_MASCARA])
        return None

    def toggle_slider(use_padding):
        return gr.update(interactive=use_padding)

    # **Asignar eventos a la interfaz**
    img.change(on_image_load, inputs=[img], outputs=[
               original_img, impainted_img])
    yolo_model_path.change(fn=upload_yolo_model,
                           inputs=yolo_model_path, outputs=None)
    detect_button.click(generate_mask_with_yolo, inputs=[img, yolo_confidence, mask_dilatation], outputs=[
                        img_yolo, processed_img, error_message_detection])
    processed_img.clear(on_clear_processed_mask, outputs=[processed_img])
    img.change(reset_mask, inputs=[img], outputs=[
               img_yolo, processed_img, final_image])
    send_button.click(process_final_image, inputs=[
                      img, processed_img, text_input, strength, guidance, steps, negative_prompt, use_padding, mask_padding_crop, enhance_faces, see_face_masks], outputs=[final_image, impainted_img, error_message_impaint])
    use_padding.change(fn=toggle_slider, inputs=use_padding,
                       outputs=mask_padding_crop)


# **Limpiar archivos previos antes de lanzar la aplicación**
delete_files([RUTA_MASCARA, RUTA_IMAGEN_FINAL])

# **Lanzar la interfaz**
demo.launch(
    #server_name="0.0.0.0",  # For deployment
    server_name = "127.0.0.1", # For testing
    server_port=7865, 
    debug=True, 
    #auth=(os.environ.get("APP_USER"), os.environ.get("APP_PASSWORD"))
)
