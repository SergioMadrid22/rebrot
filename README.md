![Logo](static/logo.png)

# Proyecto Rebrot: Restauración de Fotos DANA

Este proyecto utiliza Inteligencia Artificial para la reconstrucción y restauración de fotografías dañadas por las inundaciones de la DANA en Valencia (2024). Combina técnicas de detección de daños, segmentación precisa y generación de imágenes mediante modelos de difusión y modelos de lenguaje visual (VLM), con un enfoque especial en la **preservación de la identidad facial** y la **coherencia cromática**.

El desarrollo se enmarca en la iniciativa solidaria e institucional **"Salvem les fotos UPV / Recuperar las memorias"**.

---

## 🏛️ Arquitectura del Sistema

El proyecto está modularizado en dos capas principales para permitir la escalabilidad:

1. **Inference Server (Backend - `src/inference_server.py`):** API construida con FastAPI que gestiona la carga y descarga dinámica de modelos en la GPU para optimizar el uso de VRAM.
2. **Gradio App (Frontend - `src/gui_app.py`):** Interfaz web interactiva con soporte para edición manual (máscara editable con pincel) y procesamiento por lotes masivo (Batch).
3. **Watchdog (`src/watchdog.py`):** Sistema de monitoreo que verifica la disponibilidad del servicio cada 30 segundos, gestiona el túnel público (Loophole) y programa un reinicio preventivo diario a las 05:00 AM para evitar fugas de memoria.

---

## 📋 Requisitos Previos

* **SO:** Linux (Probado en Ubuntu 22.04 LTS).
* **Hardware:** Tarjeta NVIDIA con al menos **24 GB de VRAM** (RTX 3090, RTX 4090, A40 o superior).
* **Drivers:** CUDA 11.8 o superior.
* **Software:** Python 3.10+, `virtualenv` o `conda`.

---

## 🛠️ Instalación

Siga estos pasos para la instalación:

### 1. Descomprimir zip y crear entorno virtual
```bash
unzip rebrot-app.zip
cd rebrot-app
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configuración de entorno (`.env`)
Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:
```env
DEVICE=cuda:0
PORT_API=7890
PORT_GUI=7860
YOLO_MODEL_PATH=checkpoints/yolo.pt
HOSTNAME=rebrot-vrain
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu_usuario@gmail.com
SMTP_PASS=tu_app_password
HUGGINGFACE_HUB_TOKEN=tu_token_de_hf
```

---

## 📂 Estructura de Checkpoints

Asegúrate de colocar los pesos de los modelos entrenados en el directorio `checkpoints/`:
* `checkpoints/yolo.pt` (Detección de daños acoplada a SAM2)
* `checkpoints/segformer.pth` (Segmentación semántica con SegFormer-B0)
* `checkpoints/unet_segmentationlosspeopleandfn.pth` (Segmentación con U-Net y loss ponderada)

---

## 🚀 Ejecución

El script de inicio lanza el **Watchdog** en segundo plano, el cual levanta automáticamente el backend de inferencia, la interfaz gráfica de Gradio y expone la URL pública mediante túnel seguro (Loophole):

```bash
chmod +x iniciar_app.sh detener_app.sh
./iniciar_app.sh
```

Para detener todos los servicios vinculados:
```bash
./detener_app.sh
```

*Puedes consultar el estado y logs en tiempo real dentro del directorio `logs/`.*

---

## 🤖 Modelos Disponibles

* **Qwen-Image-Edit:** Edición global basada en instrucciones de texto. Entiende el contexto semántico completo de la imagen y no requiere de máscaras manuales de daño.
* **Stable Diffusion XL 1.0 Inpainting:** Inpainting latente de alta fidelidad. Ideal para reconstrucciones locales y fondos acotados por máscara.
* **FLUX.1-Fill-dev:** Inpainting basado en Diffusion Transformers (DiT) en cuantización NF4 de 4 bits.
* **YOLOv8 + SAM2 / SegFormer / U-Net:** Módulos de visión por computador para la detección y segmentación automática de barro, grietas y humedad.
* **SAM3 / LangSAM:** Segmentación semántica facial y de cabello para el subsistema de preservación de identidad.

---

## 🤝 Agradecimientos y Financiación

Este proyecto ha sido desarrollado en el marco de la iniciativa solidaria **"Salvem les fotos UPV / Recuperar las memorias"**, impulsada tras las inundaciones provocadas por la DANA de octubre de 2024 en la provincia de Valencia.

<p align="center">
  <img src="static/banner_agradecimientos.jpg" alt="Agradecimientos y Financiación" width="850"/>
</p>

### Entidades Financiadoras y Colaboradoras:

* **Ministerio de Cultura - Gobierno de España:** Actividad realizada con la ayuda de la Dirección General de Patrimonio Cultural y Bellas Artes (Subdirección General de Artes Visuales y Creación Contemporánea), correspondiente a la *Convocatoria de ayudas para la promoción de la fotografía 2025* (**Nº Expte: 84/357**).
* **ValgrAI (Valencian Graduate School and Research Network of Artificial Intelligence):** Por su respaldo y apoyo en el fomento y desarrollo de tecnologías basadas en Inteligencia Artificial en la Comunitat Valenciana.
* **VRAIN (Valencian Research Institute for Artificial Intelligence):** Por la dirección científica, infraestructura computacional y nodos de supercomputación GPU.
* **Universitat Politècnica de València (UPV):** A través de la **Facultad de Bellas Artes de San Carlos** y el **Instituto de Restauración del Patrimonio (IRP)**, por el suministro del archivo fotográfico, asesoramiento técnico en conservación y validación perceptual de los resultados.
* **FUJIFILM:** Por su colaboración técnica y material en los procesos de digitalización fotográfica.

## 📜 License

This project is licensed under the MIT License. For more details, see the `LICENSE` file. 🚀
