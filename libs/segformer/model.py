import os
import torch
import torch.nn as nn
import numpy as np
import gc
from PIL import Image
from torch.nn.functional import relu
from torchvision import transforms as T
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import json

with open("secret.json", "r") as f:
    secret = json.load(f)


class SegformerDANA(nn.Module):
    def __init__(self, token):
        super(SegformerDANA, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", token=token)
        self.up = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(150, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, pixel_values):
        b, c, h, w = pixel_values.size()
        output = self.model(pixel_values=pixel_values)
        logits = output.logits
        logits = self.up(logits)
        logits = self.conv(logits)
        return logits.reshape((b, 1, 1024, 1024))

CHECKPOINT_PATH = "checkpoints/segformer.pth"

class SegformerInference:
    def __init__(self, device: str):
        torch.cuda.empty_cache()
        gc.collect()
        self.predictor = SegformerDANA(token=secret["hf"])
        self.predictor.load_state_dict(
            torch.load(CHECKPOINT_PATH, weights_only=True, map_location=device))
        self.predictor = self.predictor.to(device)
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", token=secret["hf"])
        print("Modelo SegFormer cargado ?")

    def get_mask(self, image: np.ndarray):
        # Convertir la imagen numpy a PIL
        image = Image.fromarray(image).convert('RGB')

        # Aplicar transformaciones y mover al dispositivo
        image_tensor = self.processor(images=image.resize(
            (512, 512)), return_tensors="pt").pixel_values.reshape((-1, 3, 512, 512)).to(self.device)

        # Predecir la mascara
        mask = self.predictor(pixel_values=image_tensor)
        mask = torch.sigmoid(mask)

        # Binarizar la mascara
        mask_bin = (mask >= 0.5).float()  # Threshold 0.5

        # Convertir de vuelta a imagen PIL
        mask_image = T.functional.to_pil_image(mask_bin[0]).convert("RGB")
        return np.array(mask_image)
