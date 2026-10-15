#!/bin/bash
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "$(dirname "$0")"
# Directorios
VENV_PYTHON="./venv/bin/python3"
LOG_DIR="logs"

# 1. Crear carpeta de logs si no existe
mkdir -p $LOG_DIR
rm $LOG_DIR/*.log 2>/dev/null

# 2. Limpieza total para empezar de cero
echo "Limpiando procesos previos..."
pkill -f watchdog.py
pkill -f inference_server.py
pkill -f gui_app.py
pkill -f loophole

# 3. Lanzar el WATCHDOG (Watchdog)
# El watchdog se encargará de lanzar inference_server, gui_app y loophole
echo "Lanzando el Watchdog..."
nohup $VENV_PYTHON -u watchdog.py > $LOG_DIR/watchdog.log 2>&1 &
WATCHDOG_PID=$!

echo "--------------------------------------------------------"
echo "Watchdog iniciado en segundo plano (PID: $WATCHDOG_PID)."
echo "Puedes ver los logs en $LOG_DIR/"

echo "--------------------------------------------------------"