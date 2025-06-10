import socket
import time
import math
import os
import csv
from datetime import datetime
import pygame
import importlib
from coltrols.keyboard_controls import get_keyboard_controls
from coltrols.joystick_controls import get_joystick_controls

# Configurazione indirizzo server
SERVER_IP = "localhost"
SERVER_PORT = 3001
BUFFER_SIZE = 2048

# Creazione socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 0))  # Bind a una porta casuale
client_port = sock.getsockname()[1]
print(f"In ascolto sulla porta: {client_port}")

# Inizializzazione connessione con attesa TORCS
init_msg = f"SCR 1.1\n"
connected = False
print("Attendo che TORCS sia disponibile...")

while not connected:
    try:
        sock.sendto(init_msg.encode(), (SERVER_IP, SERVER_PORT))
        sock.settimeout(2.0)  # Timeout breve per la risposta
        data, addr = sock.recvfrom(BUFFER_SIZE)
        if b'TORCS' in data or data:  # Puoi personalizzare il controllo in base alla risposta attesa
            connected = True
            print("Connessione a TORCS stabilita!")
        else:
            print("Nessuna risposta da TORCS, ritento...")
    except socket.timeout:
        print("Nessuna risposta da TORCS, ritento...")
    except Exception as e:
        print(f"Errore durante la connessione: {e}")
    time.sleep(1)

sock.settimeout(None)  # Torna al comportamento di default

controls = {
    'throttle': 0.0,
    'brake': 0.0,
    'steer': 0.0,
    'gear': 0
}

def setup_csv_logging():
    # Crea la directory se non esiste
    data_dir = "torcs_training_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Creata cartella {data_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(data_dir, f"wippy_telemetry_{timestamp}.csv")
    
    # Open file and write headers
    with open(csv_filename, 'w', newline='') as csvfile:
        # Start with basic fields
        fieldnames = ['timestamp', 'angle', 'curLapTime', 'damage', 'distFromStart', 
                     'distRaced', 'fuel', 'gear', 'rpm', 'speedX', 'speedY', 'speedZ', 
                     'trackPos', 'throttle', 'brake', 'steer', 'lastLapTime', 'z']
        # Add track sensors (19 values)
        for i in range(19):
            fieldnames.append(f'track_{i}')
        
        # Add wheelSpinVel (4 values)
        for i in range(4):
            fieldnames.append(f'wheelSpinVel_{i}')
            
        # Add focus (5 values)
        for i in range(5):
            fieldnames.append(f'focus_{i}')
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return csv_filename

def log_telemetry(csv_filename, telemetry):
    with open(csv_filename, 'a', newline='') as csvfile:
        # Start with basic fields
        fieldnames = ['timestamp', 'angle', 'curLapTime', 'damage', 'distFromStart', 
                     'distRaced', 'fuel', 'gear', 'rpm', 'speedX', 'speedY', 'speedZ', 
                     'trackPos', 'throttle', 'brake', 'steer', 'lastLapTime', 'z']
        
        # Add track sensors (19 values)
        for i in range(19):
            fieldnames.append(f'track_{i}')
            
        # Add wheelSpinVel (4 values)
        for i in range(4):
            fieldnames.append(f'wheelSpinVel_{i}')
            
        # Add focus (5 values)
        for i in range(5):
            fieldnames.append(f'focus_{i}')
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Create row with current timestamp
        row = {'timestamp': datetime.now().isoformat()}
        
        # Add telemetry values
        for key in telemetry:
            if key in fieldnames and key not in ['opponents', 'racePos', 'track', 'wheelSpinVel', 'focus']:
                row[key] = telemetry[key]
        
        # Add track array values
        if 'track' in telemetry:
            for i, val in enumerate(telemetry['track']):
                if i < 19:
                    row[f'track_{i}'] = val
        
        # Add wheelSpinVel array values
        if 'wheelSpinVel' in telemetry:
            for i, val in enumerate(telemetry['wheelSpinVel']):
                if i < 4:
                    row[f'wheelSpinVel_{i}'] = val
                    
        # Add focus array values
        if 'focus' in telemetry:
            for i, val in enumerate(telemetry['focus']):
                if i < 5:
                    row[f'focus_{i}'] = val
        
        # Add current control values
        row['throttle'] = controls['throttle']
        row['brake'] = controls['brake']
        row['steer'] = controls['steer']
        
        writer.writerow(row)

def parse_telemetry(data):
    """Analizza i dati telemetrici TORCS/SCR nel formato (key value)"""
    try:
        # Remove trailing null byte if present
        if data.endswith('\x00'):
            data = data[:-1]
            
        # Create a dictionary to store values
        telemetry = {}
        
        # Extract (key value) pairs
        pairs = data.replace(')(', ')|(').split('|')
        for pair in pairs:
            # Remove outer parentheses
            pair = pair.strip('()')
            if not pair:
                continue
                
            # Split into key and value
            parts = pair.split(None, 1)  # Split on first whitespace
            if len(parts) != 2:
                continue
                
            key, value = parts
            
            # Handle arrays (space-separated values)
            if key in ['opponents', 'track', 'focus', 'wheelSpinVel']:
                telemetry[key] = [float(v) for v in value.split()]
            elif key in ['gear', 'racePos']:
                telemetry[key] = int(value)
            else:
                telemetry[key] = float(value)
                
        # Only return if we have essential data
        if 'trackPos' in telemetry:
            return telemetry
        return None
        
    except Exception as e:
        print(f"Errore parsing: {e}")
        return None

def drive_controller(telemetry):
    """Controller con input da tastiera"""
    # Format commands for SCR
    return f"(accel {controls['throttle']:.2f})(brake {controls['brake']:.2f})(gear {controls['gear']})(steer {controls['steer']:.2f})"

csv_filename = setup_csv_logging()

print(f"Logging telemetry to {csv_filename}")
print("Controlli tastiera:")
print("W - Accelera")
print("S - Frena")
print("A - Sterza a sinistra")
print("D - Sterza a destra")
print("Q - Scala marcia")
print("E - Aumenta marcia")
print("R - Resetta controlli")
print("ESC - Esci")

# Inizializzazione joystick
pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(1)
joystick.init()

print(f"Joystick: {joystick.get_name()}")

prev_gear_up = False
prev_gear_down = False

while True:
    try:
        # Ricezione dati telemetrici
        data, addr = sock.recvfrom(BUFFER_SIZE)
        if not data:
            print("\nFine partita: nessun dato ricevuto.")
            break

        telemetry = parse_telemetry(data.decode())
        if not telemetry:
            print("\nFine partita o dati non validi.")
            break

        log_telemetry(csv_filename, telemetry)

        # --- SCEGLI QUALE MODALITÀ USARE: tastiera o joystick ---
        try:
            controls, prev_gear_up, prev_gear_down = get_joystick_controls(controls, joystick, prev_gear_up, prev_gear_down)
        except Exception as e:
            print(f"\n[ERRORE JOYSTICK] {e}. Provo a mantenere i controlli precedenti.")
            pass
        # Se vuoi usare la tastiera, decommenta la riga sotto e commenta quella sopra:
        # controls = get_keyboard_controls(controls)

        # Calcolo azioni di guida
        command = drive_controller(telemetry)
        sock.sendto(command.encode(), (SERVER_IP, SERVER_PORT))

        # Visualizzazione barra freno (orizzontale) e sterzo (verticale)
        brake_bar_len = 20
        brake_pos = int(controls['brake'] * brake_bar_len)
        brake_bar = "[" + "#" * brake_pos + "-" * (brake_bar_len - brake_pos) + "]"

        steer_bar_len = 21
        steer_center = steer_bar_len // 2
        steer_pos = int(controls['steer'] * (steer_center))
        steer_bar = [" "] * steer_bar_len
        steer_bar[steer_center + steer_pos] = "|"
        steer_bar_str = "".join(steer_bar)

        print(f"\rFreno {brake_bar}  Sterzo {steer_bar_str}", end="")

    except Exception as e:
        print(f"\nErrore durante la ricezione dati: {e}")
        import traceback
        traceback.print_exc()
        break
    
    finally:
        # RIMUOVI: sock.close() e break da qui
        pass

sock.close()
print("Logging interrotto e socket chiuso.")
