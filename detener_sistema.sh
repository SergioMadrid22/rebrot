#!/bin/bash

# Configuration (Matching launch_app.sh)
PORT_API=7890
PORT_GUI=7860

echo "🛑 Stopping Rebrot Project System..."

# 1. Kill the Loophole tunnel
echo "🔗 Closing Loophole tunnel..."
pkill -f loophole

# 2. Kill the Gradio GUI
echo "🖥️  Stopping GUI Gradio (Port $PORT_GUI)..."
# Try killing by port first
fuser -k $PORT_GUI/tcp 2>/dev/null
# Then by process name just in case
pkill -f gui_app.py

# 3. Kill the Inference Server
echo "🧠 Stopping Inference Server (Port $PORT_API)..."
# Try killing by port first
fuser -k $PORT_API/tcp 2>/dev/null
# Then by process name (Crucial for freeing up GPU VRAM)
pkill -f inference_server.py
pkill -f watchdog.py

# 4. Cleanup (Optional)
# If you want to clear the logs when stopping, uncomment the next line:
#rm logs/*.log

echo "---------------------------------------"
echo "✅ All processes terminated."
echo "   Ports $PORT_GUI and $PORT_API are now free."