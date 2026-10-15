import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from diffusers.utils import load_image
from diffusers import (
    ControlNetModel, 
    StableDiffusionXLControlNetInpaintPipeline,
    AutoPipelineForInpainting
)
from transformers import pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

MODEL_IMG_DIM = 1024
SEED = 42

class SDImpainting:
    def __init__(self, device: str, use_controlnet: bool = False):
        self.device = device
        self.use_controlnet = use_controlnet
        self.generator = torch.Generator(device=device)

        # Seleccionar dtype según el dispositivo (float16 solo en CUDA)
        is_cuda = "cuda" in str(device).lower()
        torch_dtype = torch.float16 if is_cuda else torch.float32
        variant = "fp16" if is_cuda else None

        if not use_controlnet:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch_dtype,
                variant=variant,
                use_safetensors=True,
            ).to(device)
            print(f"Modelo SDXL Inpainting estándar cargado en {device} ({torch_dtype}) ✅")
        else:
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch_dtype,
                variant=variant,
                use_safetensors=True
            ).to(device)

            self.depth_estimator = pipeline(
                "depth-estimation", 
                model="Intel/dpt-hybrid-midas",
                device=device
            )

            self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                controlnet=self.controlnet,
                torch_dtype=torch_dtype,
                variant=variant,
                use_safetensors=True
            ).to(device)
            print(f"Modelo SDXL + ControlNet Depth cargado en {device} ({torch_dtype}) ✅")

    def get_depth_image(self, image):
        """Genera un mapa de profundidad para guiar a ControlNet"""
        depth_map = self.depth_estimator(image)['depth']
        depth_map = np.array(depth_map)
        
        # Normalizar a 0-255
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255.0
        depth_map = depth_map.astype(np.uint8)
        depth_pil = Image.fromarray(depth_map).convert("RGB")
        return depth_pil

    def impaint(
            self,
            image_path: str,
            mask_path: str,
            prompt: str,
            negative_prompt: str,
            strength: float,
            guidance: float,
            steps: int,
            padding_mask_crop: int = None,
            controlnet_scale: float = 0.5
    ):
        self.generator.manual_seed(SEED)
        original_image = load_image(image_path)
        original_mask = load_image(mask_path)
        orig_width, orig_height = original_image.size

        if original_mask.size != original_image.size:
            print(f"⚠️ Redimensionando máscara de {original_mask.size} a {original_image.size}")
            original_mask = original_mask.resize(original_image.size, Image.NEAREST)

        # 1. Escalado con padding manteniendo aspecto
        scale = min(MODEL_IMG_DIM / orig_width, MODEL_IMG_DIM / orig_height)
        new_size = (int(orig_width * scale), int(orig_height * scale))

        resized_image = original_image.resize(new_size, Image.LANCZOS)
        resized_mask = original_mask.resize(new_size, Image.NEAREST)

        canvas_image = Image.new("RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (255, 255, 255))
        canvas_mask = Image.new("RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (0, 0, 0))

        paste_x = (MODEL_IMG_DIM - new_size[0]) // 2
        paste_y = (MODEL_IMG_DIM - new_size[1]) // 2

        canvas_image.paste(resized_image, (paste_x, paste_y))
        canvas_mask.paste(resized_mask, (paste_x, paste_y))

        # 2. Argumentos de inferencia
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": canvas_image,
            "mask_image": canvas_mask,
            "guidance_scale": guidance,
            "strength": strength,
            "generator": self.generator,
            "num_inference_steps": steps,
            "height": MODEL_IMG_DIM,
            "width": MODEL_IMG_DIM,
        }

        if self.use_controlnet:
            depth_image = self.get_depth_image(canvas_image)
            kwargs["control_image"] = depth_image
            kwargs["controlnet_conditioning_scale"] = controlnet_scale
        else:
            if padding_mask_crop is not None:
                kwargs["padding_mask_crop"] = padding_mask_crop

        # 3. Generación
        result_canvas = self.pipe(**kwargs).images[0]
        result_crop = result_canvas.crop((paste_x, paste_y, paste_x + new_size[0], paste_y + new_size[1]))
        result_crop = result_crop.resize((orig_width, orig_height), Image.LANCZOS)

        return result_crop

        # 4. POSTPROCESAMIENTO: Preservación tonal en espacio LAB
        original_np = np.array(original_image)
        generated_np = np.array(result_crop)
        
        src_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB).astype("float32")
        tgt_lab = cv2.cvtColor(generated_np, cv2.COLOR_RGB2LAB).astype("float32")

        mask_np = np.array(original_mask.convert("L"))
        mask_generated = (mask_np > 128).astype(np.uint8)
        mask_clean = (mask_np <= 128).astype(np.uint8)

        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(mask_generated, kernel, iterations=1)
        context_mask = ((dilated == 1) & (mask_clean == 1))

        # Transferencia estadística (Reinhard) guiada por el contexto sano
        if np.any(context_mask):
            for i in range(3):  # Canales L, A, B
                mu_src, std_src = src_lab[context_mask, i].mean(), src_lab[context_mask, i].std()
                mu_tgt, std_tgt = tgt_lab[mask_generated == 1, i].mean(), tgt_lab[mask_generated == 1, i].std()

                tgt_lab[mask_generated == 1, i] = ((tgt_lab[mask_generated == 1, i] - mu_tgt) * (std_src / (std_tgt + 1e-6))) + mu_src

        tgt_corrected = cv2.cvtColor(np.clip(tgt_lab, 0, 255).astype("uint8"), cv2.COLOR_LAB2RGB)

        # 5. MEZCLADO SUAVE (Feathering)
        mask_pil = original_mask.convert("L")
        soft_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=15))
        soft_mask_np = np.array(soft_mask).astype(np.float32) / 255.0
        soft_mask_np = np.expand_dims(soft_mask_np, axis=-1)

        # Composición final
        final_np = (tgt_corrected.astype(np.float32) * soft_mask_np) + \
                   (original_np.astype(np.float32) * (1.0 - soft_mask_np))
        
        final_result = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
        return final_result
        
        
        # ---ALTERNATIVA 0
        # Convertir imágenes a OpenCV y HSV
        src = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        tgt = cv2.cvtColor(np.array(result_crop), cv2.COLOR_RGB2BGR)

        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2HSV)

        h_src, s_src, v_src = cv2.split(hsv_src)
        h_tgt, s_tgt, v_tgt = cv2.split(hsv_tgt)

        # Convertir la máscara binaria
        mask_np = np.array(original_mask.convert("L"))

        # Crear máscara de zonas generadas (255) y limpias (0)
        mask_generated = (mask_np == 255).astype(np.uint8)
        mask_clean = (mask_np == 0).astype(np.uint8)

        # Crear zona de contexto (bordes alrededor de las manchas)
        kernel = np.ones((15, 15), np.uint8)  # tamaño del vecindario a considerar
        dilated = cv2.dilate(mask_generated, kernel, iterations=1)

        # Zona de contexto = dilatada menos las manchas (y solo si es limpia)
        context_mask = ((dilated == 1) & (mask_clean == 1))

        # Usar zona de contexto para obtener el brillo y saturación de referencia
        mean_s_src = np.mean(s_src[context_mask])
        mean_v_src = np.mean(v_src[context_mask])

        # Medias en las zonas generadas
        generated_mask_bool = mask_generated.astype(bool)
        mean_s_tgt = np.mean(s_tgt[generated_mask_bool])
        mean_v_tgt = np.mean(v_tgt[generated_mask_bool])

        # Corrección con suavizado
        s_weight = 0.7
        v_weight = 0.7

        s_tgt_corrected = s_tgt * (mean_s_src / (mean_s_tgt + 1e-6))
        v_tgt_corrected = v_tgt * (mean_v_src / (mean_v_tgt + 1e-6))

        # Aplicar el cambio **solo en las zonas generadas**
        s_tgt[generated_mask_bool] = np.clip(
            s_weight * s_tgt_corrected[generated_mask_bool] + (1 - s_weight) * s_tgt[generated_mask_bool],
           0, 255
        ).astype(np.uint8)
        
        v_tgt[generated_mask_bool] = np.clip(
           v_weight * v_tgt_corrected[generated_mask_bool] + (1 - v_weight) * v_tgt[generated_mask_bool],
            0, 255
        ).astype(np.uint8)

        # Reconstrucción final
        hsv_matched = cv2.merge([h_tgt, s_tgt, v_tgt])
        bgr_matched = cv2.cvtColor(hsv_matched, cv2.COLOR_HSV2BGR)
        result_crop = Image.fromarray(cv2.cvtColor(bgr_matched, cv2.COLOR_BGR2RGB))

        original_image_np = np.array(original_image)
        result_crop_np = np.array(result_crop)
        mask_np = np.array(original_mask)

        final_np = original_image_np.copy()
        white_mask = mask_np == 255
        final_np[white_mask] = result_crop_np[white_mask]
        final_result = Image.fromarray(final_np)
        return final_result
        return result_crop

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
        return result_crop