#!/bin/bash

# ================= CONFIGURACIÓN =================
TARGET_USER="salvem"
APP_SCRIPT="batch_processor.py"
PORT=7860
# =================================================

echo "🛑 Iniciando parada segura para el usuario: $TARGET_USER"

# 1. Matar el script de Python (Gradio/Modelos)
# -u : Restringe la búsqueda al usuario específico
# -f : Busca en toda la línea de comandos (no solo el nombre del proceso)
if pgrep -u "$TARGET_USER" -f "$APP_SCRIPT" > /dev/null; then
    pkill -u "$TARGET_USER" -f "$APP_SCRIPT"
    echo "✅ Proceso Python ($APP_SCRIPT) de $TARGET_USER detenido."
else
    echo "ℹ️  No se encontró el script Python corriendo para $TARGET_USER."
fi

# 2. Matar túneles (Solo los de salvem)
for TUNNEL in "loophole" "cloudflared" "ngrok"; do
    if pgrep -u "$TARGET_USER" -f "$TUNNEL" > /dev/null; then
        pkill -u "$TARGET_USER" -f "$TUNNEL"
        echo "✅ Túnel $TUNNEL de $TARGET_USER cerrado."
    fi
done

# 3. Liberar el puerto 7860 (Solo si lo ocupa salvem)
# lsof flags:
# -i : puerto
# -u : usuario
# -a : AND lógico (debe cumplir SER el puerto Y SER el usuario)
# -t : Terse (solo devuelve el PID)
PIDS=$(lsof -i :$PORT -u "$TARGET_USER" -a -t)

if [ -n "$PIDS" ]; then
    kill -9 $PIDS
    echo "✅ Puerto $PORT liberado (PIDs: $PIDS pertenecían a $TARGET_USER)."
else
    echo "✅ El puerto $PORT ya estaba libre (o no pertenece a $TARGET_USER)."
fi

echo "¡Limpieza completada!"