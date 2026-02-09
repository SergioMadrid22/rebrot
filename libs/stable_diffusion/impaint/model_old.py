import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from utils import fill_little_spaces
from diffusers.utils import load_image
from diffusers import AutoPipelineForInpainting
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../..")))
# from diffusers import StableDiffusionInpaintPipeline

MODEL_IMG_DIM = 1024


class SDImpainting:
    def __init__(self, device: str):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        ).to(device)

        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        # "runwayml/stable-diffusion-inpainting",
        # torch_dtype=torch.float16,
        # token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        # ).to(device)

        self.generator = torch.Generator(device=device).manual_seed(0)

        print("Modelo Stable Diffusion XL 1.0 descargado ✅")
        # print("Modelo Stable Diffusion 1.5 descargado ✅")

    def impaint(
            self,
            image_path: str,
            mask_path: str,
            prompt: str,
            negative_prompt: str,
            strength: float,
            guidance: float,
            steps: int,
            padding_mask_crop: int,
    ):
        # Carga las imágenes originales
        original_image = load_image(image_path)
        original_mask = load_image(mask_path)

        # Guarda las dimensiones originales
        orig_width, orig_height = original_image.size

        # Calcula el factor de escala para ajustar la imagen al canvas de 1024x1024 sin distorsionar la relación de aspecto
        scale = min(MODEL_IMG_DIM / orig_width, MODEL_IMG_DIM / orig_height)
        new_size = (int(orig_width * scale), int(orig_height * scale))

        # Redimensiona la imagen y la máscara manteniendo la relación de aspecto
        resized_image = original_image.resize(new_size, Image.LANCZOS)
        resized_mask = original_mask.resize(new_size, Image.LANCZOS)

        # Crea un canvas de 1024x1024 para la imagen y la máscara
        canvas_image = Image.new(
            "RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (255, 255, 255))
        # Ajusta el color de fondo de la máscara según necesites
        canvas_mask = Image.new(
            "RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (0, 0, 0))

        # Calcula las posiciones para centrar la imagen redimensionada en el canvas
        paste_x = (MODEL_IMG_DIM - new_size[0]) // 2
        paste_y = (MODEL_IMG_DIM - new_size[1]) // 2

        canvas_image.paste(resized_image, (paste_x, paste_y))
        canvas_mask.paste(resized_mask, (paste_x, paste_y))

        # Ejecuta el pipeline de inpainting en el canvas
        # Crear kwargs opcionalmente
        extra_args = {}
        if padding_mask_crop is not None:
            extra_args["padding_mask_crop"] = padding_mask_crop

        result_canvas = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canvas_image,
            mask_image=canvas_mask,
            guidance_scale=guidance,
            strength=strength,
            generator=self.generator,
            num_inference_steps=steps,
            **extra_args
        ).images[0]

        # Recorta la parte central que contiene la imagen original redimensionada
        result_crop = result_canvas.crop(
            (paste_x, paste_y, paste_x + new_size[0], paste_y + new_size[1]))
        # Reescala la imagen resultante a las dimensiones originales
        result_crop = result_crop.resize(
            (orig_width, orig_height), Image.LANCZOS)

        # if see_face_masks:
        #     # Asegúrate de que result_crop esté en modo RGBA
        #     result_crop = result_crop.convert("RGBA")

        #     # Convertimos face_mask a imagen PIL en modo 'L'
        #     face_mask_img = Image.fromarray(
        #         face_mask.astype(np.uint8), mode='L')

        #     # Invertimos la máscara si es necesario (según cómo sea tu detector)
        #     # Asegura que la máscara tenga buen rango
        #     face_mask_img = ImageOps.autocontrast(face_mask_img)

        #     # Creamos una imagen blanca del mismo tamaño
        #     white_image = Image.new(
        #         "RGBA", result_crop.size, (255, 255, 255, 255))

        #     # Componemos: donde la máscara es blanca, se usa white_image; en el resto, se usa result_crop
        #     result_crop = Image.composite(
        #         white_image, result_crop, face_mask_img)
        # elif keep_faces:
        #     # Asegúrate de que ambas imágenes estén en modo RGBA
        #     result_crop = result_crop.convert("RGBA")
        #     original_image_rgba = original_image.convert("RGBA")

        #     # Convertir face_mask a imagen PIL y asegurarse que tenga valores 0-255
        #     face_mask_img = Image.fromarray(
        #         face_mask.astype(np.uint8), mode='L')
        #     face_mask_img = ImageOps.autocontrast(face_mask_img)

        #     # Componer: donde la máscara es blanca, tomar de original; donde es negra, dejar el resultado
        #     result_crop = Image.composite(
        #         original_image_rgba, result_crop, face_mask_img)

        return result_crop

        # ---ALTERNATIVA 0
        # Convertir imágenes a OpenCV y HSV
        # src = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        # tgt = cv2.cvtColor(np.array(result_crop), cv2.COLOR_RGB2BGR)

        # hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # hsv_tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2HSV)

        # h_src, s_src, v_src = cv2.split(hsv_src)
        # h_tgt, s_tgt, v_tgt = cv2.split(hsv_tgt)

        # Convertir la máscara binaria
        # mask_np = np.array(original_mask.convert("L"))

        # Crear máscara de zonas generadas (255) y limpias (0)
        # mask_generated = (mask_np == 255).astype(np.uint8)
        # mask_clean = (mask_np == 0).astype(np.uint8)

        # Crear zona de contexto (bordes alrededor de las manchas)
        # kernel = np.ones((15, 15), np.uint8)  # tamaño del vecindario a considerar
        # dilated = cv2.dilate(mask_generated, kernel, iterations=1)

        # Zona de contexto = dilatada menos las manchas (y solo si es limpia)
        # context_mask = ((dilated == 1) & (mask_clean == 1))

        # Usar zona de contexto para obtener el brillo y saturación de referencia
        # mean_s_src = np.mean(s_src[context_mask])
        # mean_v_src = np.mean(v_src[context_mask])

        # Medias en las zonas generadas
        # generated_mask_bool = mask_generated.astype(bool)
        # mean_s_tgt = np.mean(s_tgt[generated_mask_bool])
        # mean_v_tgt = np.mean(v_tgt[generated_mask_bool])

        # Corrección con suavizado
        # s_weight = 0.7
        # v_weight = 0.7

        # s_tgt_corrected = s_tgt * (mean_s_src / (mean_s_tgt + 1e-6))
        # v_tgt_corrected = v_tgt * (mean_v_src / (mean_v_tgt + 1e-6))

        # Aplicar el cambio **solo en las zonas generadas**
        # s_tgt[generated_mask_bool] = np.clip(
        #     s_weight * s_tgt_corrected[generated_mask_bool] + (1 - s_weight) * s_tgt[generated_mask_bool],
        #    0, 255
        # ).astype(np.uint8)

        # v_tgt[generated_mask_bool] = np.clip(
        #    v_weight * v_tgt_corrected[generated_mask_bool] + (1 - v_weight) * v_tgt[generated_mask_bool],
        #     0, 255
        # ).astype(np.uint8)

        # Reconstrucción final
        # hsv_matched = cv2.merge([h_tgt, s_tgt, v_tgt])
        # bgr_matched = cv2.cvtColor(hsv_matched, cv2.COLOR_HSV2BGR)
        # result_crop = Image.fromarray(cv2.cvtColor(bgr_matched, cv2.COLOR_BGR2RGB))

        # return result_crop

        # ALTERNATIVA 1 --- Ajuste de brillo y saturación para que coincida con la imagen original ---
        # src = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        # tgt = cv2.cvtColor(np.array(result_crop), cv2.COLOR_RGB2BGR)

        # hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # hsv_tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2HSV)

        # h_src, s_src, v_src = cv2.split(hsv_src)
        # h_tgt, s_tgt, v_tgt = cv2.split(hsv_tgt)

        # Convertir la máscara binaria a NumPy
        # mask_np = np.array(original_mask.convert("L"))
        # valid_pixels = mask_np == 0  # True donde no hay mancha

        # Calcular medias solo en zonas válidas
        # mean_s_src = np.mean(s_src[valid_pixels])
        # mean_v_src = np.mean(v_src[valid_pixels])

        # Calcular medias globales de la imagen generada
        # mean_s_tgt = np.mean(s_tgt)
        # mean_v_tgt = np.mean(v_tgt)

        # Pesos para suavizar la corrección
        # s_weight = 0.7
        # v_weight = 0.7

        # Aplicar corrección parcial
        # s_tgt_corrected = s_tgt * (mean_s_src / (mean_s_tgt + 1e-6))
        # v_tgt_corrected = v_tgt * (mean_v_src / (mean_v_tgt + 1e-6))

        # s_tgt = np.clip(s_weight * s_tgt_corrected + (1 - s_weight) * s_tgt, 0, 255).astype(np.uint8)
        # v_tgt = np.clip(v_weight * v_tgt_corrected + (1 - v_weight) * v_tgt, 0, 255).astype(np.uint8)

        # Reconstruir y convertir de nuevo a PIL
        # hsv_matched = cv2.merge([h_tgt, s_tgt, v_tgt])
        # bgr_matched = cv2.cvtColor(hsv_matched, cv2.COLOR_HSV2BGR)
        # result_crop = Image.fromarray(cv2.cvtColor(bgr_matched, cv2.COLOR_BGR2RGB))

        # return result_crop

        # -----ALTERNATIVA 2
        # Convertir a arrays
        # original_np = np.array(original_image).astype(np.float32)
        # result_np = np.array(result_crop).astype(np.float32)

        # === Crear máscara difusa (feathered mask) ===
        # gray_mask = original_mask.convert("L")
        # blurred_mask = gray_mask.filter(ImageFilter.GaussianBlur(radius=15))  # Ajusta el radio si lo ves muy fuerte o débil
        # feather_mask_np = np.array(blurred_mask).astype(np.float32) / 255.0  # Escala 0-1

        # === Ajuste de brillo (basado en zonas no enmascaradas) ===
        # mask_np = np.array(gray_mask).astype(np.float32) / 255.0
        # mask_bool = mask_np > 0.5
        # inverse_mask_bool = ~mask_bool

        # def brightness(img):
        # return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        # target_brightness = brightness(original_np)[inverse_mask_bool]
        # generated_brightness = brightness(result_np)[mask_bool]

        # if generated_brightness.mean() > 1e-3:
        # brightness_factor = target_brightness.mean() / generated_brightness.mean()
        # else:
        # brightness_factor = 1.0

        # adjusted_result_np = result_np * brightness_factor
        # adjusted_result_np = np.clip(adjusted_result_np, 0, 255)

        # === Mezcla suave usando máscara difusa ===
        # final_np = (original_np * (1 - feather_mask_np[..., None]) +
        # adjusted_result_np * feather_mask_np[..., None])
        # final_np = np.clip(final_np, 0, 255)

        # return Image.fromarray(final_np.astype(np.uint8))

        # ------ALTERNATIVA 3

        # Convertir a arrays
        # original_np = np.array(result_crop).astype(np.float32)
        # result_np = np.array(result_crop).astype(np.float32)

        # === Crear máscara difusa (feathered mask) ===
        # gray_mask = original_mask.convert("L")
        # blurred_mask = gray_mask.filter(ImageFilter.GaussianBlur(radius=15))  # Ajusta el radio si lo ves muy fuerte o débil
        # feather_mask_np = np.array(blurred_mask).astype(np.float32) / 255.0  # Escala 0-1

        # === Ajuste de brillo (basado en zonas no enmascaradas) ===
        # mask_np = np.array(gray_mask).astype(np.float32) / 255.0
        # mask_bool = mask_np > 0.5
        # inverse_mask_bool = ~mask_bool

        # def brightness(img):
        #   return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        # target_brightness = brightness(original_np)[inverse_mask_bool]
        # generated_brightness = brightness(result_np)[mask_bool]

        # if generated_brightness.mean() > 1e-3:
        #   brightness_factor = target_brightness.mean() / generated_brightness.mean()
        # else:
        #   brightness_factor = 1.0

        # adjusted_result_np = result_np * brightness_factor
        # adjusted_result_np = np.clip(adjusted_result_np, 0, 255)

        # === Mezcla suave usando máscara difusa ===
        # final_np = (original_np * (1 - feather_mask_np[..., None]) +
        #           adjusted_result_np * feather_mask_np[..., None])
        # final_np = np.clip(final_np, 0, 255)

        # return Image.fromarray(final_np.astype(np.uint8))

        # ----- ALTERNATIVA 4
        # Fusiona usando la imagen original con los pixeles generados para la mascara
        original_np = np.array(original_image)
        result_np = np.array(result_crop)
        mask_np = np.array(original_mask)

        white_mask = mask_np == 255  # Solo cambia donde la máscara es blanca

        final_np = original_np.copy()
        final_np[white_mask] = result_np[white_mask]

        final_result = Image.fromarray(final_np)

        return final_result
