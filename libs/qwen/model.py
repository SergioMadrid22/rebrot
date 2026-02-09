import torch
from PIL import Image, ImageOps
from diffusers import QwenImageEditPlusPipeline

class QwenImpainting:
    def __init__(self, device):
        self.device = device
        self.model_id = "ovedrive/Qwen-Image-Edit-2511-4bit"        
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        self.pipeline.to(self.device)
        print("✅ Qwen cargado correctamente.")

    def impaint(self, image_path, mask_path, prompt, negative_prompt, strength, guidance, steps, **kwargs):
        """
        Qwen es un modelo de edición global, no un inpainter estricto.
        Para integrarlo, generamos la edición y luego usamos la máscara para 
        mezclar solo las partes cambiadas sobre la original.
        """
        
        original_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        
        prompt = (
            "Task: Professional Photo Restoration. "
            "Action: Remove all brown mud stains, dirt spots, and water damage overlays from the image. "
            "Detail: Inpaint the cleaned areas to seamlessly match the original background texture and lighting. "
            "Constraint: Strictly preserve the original facial features, identity, and skin texture of the people. "
            "Do not smooth the skin excessively. Do not change the clothing style. "
            "Quality: Sharpen details, correct color bleeding, and ensure a photorealistic, high-resolution result."
        )
        negative_prompt = (
            "mud, dirt, stains, scratches, blur, noise, "
            "cartoon, illustration, painting, 3d render, anime, "
            "distorted faces, bad anatomy, extra fingers, changed identity, "
            "makeup, plastic skin, smooth skin, ai generated look, "
            "overexposed, underexposed, color drift"
        )
        
        inputs = {
            "image": original_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generator": torch.manual_seed(42),
            "true_cfg_scale": guidance,
            "num_inference_steps": int(steps),
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs, return_dict=False)
            
        restored_image = output[0][0]
        if restored_image.size != original_image.size:
            restored_image = restored_image.resize(original_image.size, Image.LANCZOS)
            
        # Mezclar: Fondo original + Parte restaurada
        final_image = Image.composite(restored_image, original_image, mask_image)
        
        return restored_image