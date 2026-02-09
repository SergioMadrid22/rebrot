#!/bin/bash

# ================= CONFIGURACIÓN =================
# Elige el nombre que quieres para tu URL (ej: https://rebrot-vrain.loophole.site)
HOSTNAME_DESEADO="rebrot-vrain" 

# Carpeta de entorno virtual
VENV_DIR=".venv"

# Puerto de Gradio
PORT=7860
# =================================================

echo "🔧 Iniciando despliegue con URL Permanente (Loophole)..."

# 1. Limpieza de puertos y procesos viejos
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
  kill -9 $PID
fi
pkill -f loophole
# ----------------------------------------

# 2. Activar entorno virtual
if [ -d "$VENV_DIR" ]; then
    source $VENV_DIR/bin/activate
fi

# 3. Lanzar App Gradio
echo "🚀 Lanzando main_yolo.py..."
nohup python3 -u main_yolo.py > app_output.log 2>&1 &
APP_PID=$!

# Esperar a que Gradio arranque
echo "⏳ Esperando a que Gradio inicie..."
while ! nc -z localhost $PORT; do   
  sleep 1
done
echo "✅ App online."

# 4. Lanzar Túnel Loophole con URL Fija
# Asume que el binario 'loophole' está en la misma carpeta
if [ -f "./loophole" ]; then
    echo "🌍 Publicando en: https://$HOSTNAME_DESEADO.loophole.site"
    nohup ./loophole http $PORT --hostname $HOSTNAME_DESEADO > tunnel_output.log 2>&1 &
else
    echo "❌ Error: No encuentro el archivo './loophole'. Descárgalo primero."
fi

echo "========================================================"
echo "🎉 URL PERMANENTE: https://$HOSTNAME_DESEADO.loophole.site"
echo "========================================================"