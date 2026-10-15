import torch
import cv2
import numpy as np
import os
from PIL import Image, ImageFilter, ImageOps
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from transformers import BitsAndBytesConfig

MODEL_IMG_DIM = 1024

class FluxImpainting:
    def __init__(self, device: str):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        print("SERVIDOR: Cargando Transformer de Flux en 4-bit...")
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )

        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", 
            transformer=transformer,
            torch_dtype=torch.bfloat16, 
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        ).to(device)

        print("Modelo Flux Impainting (4-bit NF4) cargado ✅")

    def impaint(self, image_path: str, mask_path: str, prompt: str, negative_prompt: str, strength: float, guidance: float, steps: int):
        # 1. Carga de imágenes originales
        original_image = ImageOps.exif_transpose(load_image(image_path)).convert("RGB")
        original_mask = ImageOps.exif_transpose(load_image(mask_path)).convert("L")
        orig_width, orig_height = original_image.size

        if original_mask.size != original_image.size:
            print(f"⚠️ Redimensionando máscara de {original_mask.size} a {original_image.size}")
            original_mask = original_mask.resize(original_image.size, Image.NEAREST)

        # 2. Lógica de escalado con padding (Canvas 1024x1024)
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

        # 3. Inferencia de Flux
        result_canvas = self.pipe(
            prompt=prompt,
            image=canvas_image,
            mask_image=canvas_mask,
            height=MODEL_IMG_DIM,
            width=MODEL_IMG_DIM,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=512,
            generator=torch.Generator().manual_seed(42)
        ).images[0]

        # 4. Recortar y redimensionar a tamaño original
        result_crop = result_canvas.crop((paste_x, paste_y, paste_x + new_size[0], paste_y + new_size[1]))
        result_crop = result_crop.resize((orig_width, orig_height), Image.LANCZOS)

        # --- POST-PROCESAMIENTO AVANZADO (METODOLOGÍA SDXL) ---

        # 1. Convertir a NumPy/OpenCV en espacio LAB para igualar color/iluminación
        original_np = np.array(original_image)
        generated_np = np.array(result_crop)
        
        src_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB).astype("float32")
        tgt_lab = cv2.cvtColor(generated_np, cv2.COLOR_RGB2LAB).astype("float32")

        # 2. Definir áreas de contexto (zonas sanas cercanas a la mancha)
        mask_np = np.array(original_mask)
        mask_generated = (mask_np > 128).astype(np.uint8)
        mask_clean = (mask_np <= 128).astype(np.uint8)

        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(mask_generated, kernel, iterations=1)
        context_mask = ((dilated == 1) & (mask_clean == 1))

        # 3. Transferencia de color si hay contexto disponible
        if np.any(context_mask):
            for i in range(3): # Canales L, A, B
                mu_src, std_src = src_lab[context_mask, i].mean(), src_lab[context_mask, i].std()
                mu_tgt, std_tgt = tgt_lab[mask_generated == 1, i].mean(), tgt_lab[mask_generated == 1, i].std()

                # Ajustar píxeles generados para que coincidan con la estadística del contexto
                tgt_lab[mask_generated == 1, i] = ((tgt_lab[mask_generated == 1, i] - mu_tgt) * (std_src / (std_tgt + 1e-6))) + mu_src

        # 4. Volver a RGB y limitar valores
        tgt_corrected = cv2.cvtColor(np.clip(tgt_lab, 0, 255).astype("uint8"), cv2.COLOR_LAB2RGB)

        # 5. Mezclado suave (Feathering) para eliminar bordes duros
        soft_mask = original_mask.filter(ImageFilter.GaussianBlur(radius=15))
        soft_mask_np = np.array(soft_mask).astype(np.float32) / 255.0
        soft_mask_np = np.expand_dims(soft_mask_np, axis=-1) # (H, W, 1)

        # 6. Composición final (Alpha Blending)
        # Resultado = (Corregido * MáscaraSuave) + (Original * (1 - MáscaraSuave))
        final_np = (tgt_corrected.astype(np.float32) * soft_mask_np) + \
                   (original_np.astype(np.float32) * (1.0 - soft_mask_np))
        
        final_result = Image.fromarray(np.clip(final_np, 0, 255).astype(np.uint8))
        
        return final_result