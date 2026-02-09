#!/bin/bash

VENV_DIR=".venv"
PORT_API=7890
PORT_GUI=7860

# 1. Limpieza
fuser -k $PORT_API/tcp
fuser -k $PORT_GUI/tcp
pkill -f loophole

# 2. Activar entorno
source $VENV_DIR/bin/activate

# 3. Lanzar Servidor de Inferencia (Backend)
echo "🧠 Iniciando Servidor de Inferencia (GPU)..."
nohup python3 -u inference_server.py > logs/server.log 2>&1 &
SERVER_PID=$!

# Esperar a que el servidor responda
echo "⏳ Esperando al servidor de inferencia..."
while ! nc -z localhost $PORT_API; do sleep 1; done
echo "✅ Servidor de Inferencia ONLINE."

# 4. Lanzar Interfaz Gráfica (Frontend)
echo "🖥️ Iniciando GUI Gradio..."
nohup python3 -u gui_app.py > logs/gui.log 2>&1 &

# Esperar a GUI
while ! nc -z localhost $PORT_GUI; do sleep 1; done

# 5. Túnel
echo "🌍 Abriendo túnel Loophole..."
./loophole http $PORT_GUI --hostname rebrot-vrain > logs/loophole.log 2>&1 &