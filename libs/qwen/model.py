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
        ).to(self.device)
        #self.pipeline.to(self.device)
        print("✅ Qwen cargado correctamente.")

    def impaint(self, image_path, mask_path, prompt, negative_prompt, strength, guidance, steps, **kwargs):
        """
        Qwen es un modelo de edición global, no un inpainter estricto.
        Para integrarlo, generamos la edición y luego usamos la máscara para 
        mezclar solo las partes cambiadas sobre la original.
        """
        
        original_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        
        inputs = {
            "image": original_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "generator": torch.Generator(device=self.device).manual_seed(42),
            "true_cfg_scale": guidance,
            "num_inference_steps": int(steps),
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            
        restored_image = output.images[0]
        if restored_image.size != original_image.size:
            restored_image = restored_image.resize(original_image.size, Image.LANCZOS)
        
        return restored_image