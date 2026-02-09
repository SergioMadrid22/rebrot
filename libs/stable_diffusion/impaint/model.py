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
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

MODEL_IMG_DIM = 1024

class SDImpainting:
    def __init__(self, device: str, use_controlnet: bool = False):
        self.device = device
        self.use_controlnet = use_controlnet
        self.generator = torch.Generator(device=device).manual_seed(0)

        if not use_controlnet:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(device)
            print("Modelo SDXL Inpainting estándar cargado ✅")
        else:
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch.float16,
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
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(device)
            print("Modelo SDXL + ControlNet Depth cargado ✅")

    def get_depth_image(self, image):
        """Genera un mapa de profundidad para guiar a ControlNet"""
        # El estimador devuelve un diccionario con la imagen de profundidad
        depth_map = self.depth_estimator(image)['depth']
        
        # Convertir a formato PIL RGB (3 canales) que es lo que espera el pipeline
        depth_map = np.array(depth_map)
        
        # Normalizar 
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
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
            padding_mask_crop: int,
            controlnet_scale: float = 0.5
    ):
        original_image = load_image(image_path)
        original_mask = load_image(mask_path)
        orig_width, orig_height = original_image.size

        # Lógica de escalado con padding para mantener proporciones
        scale = min(MODEL_IMG_DIM / orig_width, MODEL_IMG_DIM / orig_height)
        new_size = (int(orig_width * scale), int(orig_height * scale))

        resized_image = original_image.resize(new_size, Image.LANCZOS)
        resized_mask = original_mask.resize(new_size, Image.LANCZOS)

        canvas_image = Image.new("RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (255, 255, 255))
        canvas_mask = Image.new("RGB", (MODEL_IMG_DIM, MODEL_IMG_DIM), (0, 0, 0))

        paste_x = (MODEL_IMG_DIM - new_size[0]) // 2
        paste_y = (MODEL_IMG_DIM - new_size[1]) // 2

        canvas_image.paste(resized_image, (paste_x, paste_y))
        canvas_mask.paste(resized_mask, (paste_x, paste_y))

        # Argumentos base
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": canvas_image,
            "mask_image": canvas_mask,
            "guidance_scale": guidance,
            "strength": strength,
            "generator": self.generator,
            "num_inference_steps": steps,
        }

        if self.use_controlnet:
            depth_image = self.get_depth_image(canvas_image)
            kwargs["control_image"] = depth_image
            kwargs["controlnet_conditioning_scale"] = controlnet_scale
        else:
            if padding_mask_crop is not None:
                kwargs["padding_mask_crop"] = padding_mask_crop

        result_canvas = self.pipe(**kwargs).images[0]
        result_crop = result_canvas.crop((paste_x, paste_y, paste_x + new_size[0], paste_y + new_size[1]))
        result_crop = result_crop.resize((orig_width, orig_height), Image.LANCZOS)

        return result_crop