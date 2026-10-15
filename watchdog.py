import subprocess
import time
import requests
import os
import signal
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURACIÓN ---
VENV_PYTHON = "./venv/bin/python3"
PORT_API = os.getenv("PORT_API", 7890)
PORT_GUI = os.getenv("PORT_GUI", 7860)
HOSTNAME = os.getenv("HOSTNAME", "rebrot-vrain")
PUBLIC_URL = f"https://{HOSTNAME}.loophole.site"
LOOPHOLE_CMD = f"./loophole http {PORT_GUI} --hostname {HOSTNAME} --verbose"
LOOPHOLE_LOG = "logs/loophole.log"

# Variable para controlar que solo reinicie una vez al día
last_reboot_date = None 

def kill_process_on_port(port):
    try:
        result = subprocess.check_output(["lsof", "-t", f"-i:{port}"])
        pids = result.decode().strip().split('\n')
        for pid in pids:
            if pid:
                print(f"Matando proceso en puerto {port} (PID: {pid})...")
                os.kill(int(pid), signal.SIGKILL)
    except:
        pass

def check_http_health(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        return response.status_code in [200, 401]
    except:
        return False

def run_background(cmd, log_file):
    f = open(log_file, "a")
    return subprocess.Popen(cmd.split(), stdout=f, stderr=f, preexec_fn=os.setsid)

print("Watchdog con Reinicio Programado (05:00 AM) iniciado.")

while True:
    timestamp = datetime.now()

    # --- LÓGICA DE REINICIO PROGRAMADO (5:00 AM) ---
    # Comprobamos si son las 5:00 y si aún no hemos reiniciado hoy a esta hora
    if timestamp.hour == 5 and timestamp.minute == 0 and last_reboot_date != timestamp.date():
        print(f"{timestamp.strftime('%H:%M:%S')} - Iniciando reinicio diario preventivo...")
        
        # Matamos el túnel y la GUI para limpiar la sesión
        os.system("pkill -f loophole")
        kill_process_on_port(PORT_GUI)
        
        last_reboot_date = timestamp.date()
        print("Procesos eliminados. El watchdog los levantará en el siguiente ciclo.")
        time.sleep(60) # Esperamos a que pase el minuto 05:00 para no repetir

    # --- 1. COMPROBAR API ---
    if not check_http_health(f"http://localhost:{PORT_API}/health"):
        print(f"API caida/bloqueada. Reiniciando...")
        kill_process_on_port(PORT_API)
        run_background(f"{VENV_PYTHON} -u inference_server.py", "logs/server.log")
        time.sleep(20)

    # --- 2. COMPROBAR GUI ---
    gui_local_ok = check_http_health(f"http://localhost:{PORT_GUI}/")
    if not gui_local_ok:
        print(f"GUI caida/bloqueada. Reiniciando...")
        kill_process_on_port(PORT_GUI)
        run_background(f"{VENV_PYTHON} -u gui_app.py", "logs/gui.log")
        time.sleep(5)

    # --- 3. COMPROBAR TÚNEL ---
    tunnel_proc_check = subprocess.run(["pgrep", "-f", "loophole"], capture_output=True)
    tunnel_url_ok = check_http_health(PUBLIC_URL)
    restart_tunnel = False

    if not tunnel_proc_check.stdout:
        print("Túnel Loophole: Proceso no encontrado.")
        restart_tunnel = True
    elif gui_local_ok and not tunnel_url_ok:
        print("Túnel Loophole: GUI local funciona pero URL pública no responde (ZOMBIE).")
        restart_tunnel = True

    if restart_tunnel:
        print("Relanzando túnel Loophole...")
        os.system("pkill -9 -f loophole") # Matamos con -9 para asegurar
        time.sleep(2)
        # Limpiamos el log antes de relanzar para que no lea errores viejos
        with open(LOOPHOLE_LOG, "w") as f: f.write(f"--- Reinicio Watchdog {timestamp} ---\n")
        run_background(LOOPHOLE_CMD, LOOPHOLE_LOG)
        time.sleep(10) # Damos tiempo para conectar

    time.sleep(600) # Vigilancia cada 30 segundos