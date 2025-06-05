import socket
import time
import os
from pynput import keyboard
import csv
from datetime import datetime
# Configurazione indirizzo server
SERVER_IP = "localhost"
SERVER_PORT = 3001
BUFFER_SIZE = 2048

# Creazione socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 0))  # Bind a una porta casuale
client_port = sock.getsockname()[1]
print(f"In ascolto sulla porta: {client_port}")

# Inizializzazione connessione
init_msg = f"SCR 1.1\n"
sock.sendto(init_msg.encode(), (SERVER_IP, SERVER_PORT))
print("Inviato messaggio di inizializzazione")

controls = {
    'throttle': 0.0,
    'brake': 0.0,
    'steer': 0.0,
    'gear': 1  # Neutral = 0, First = 1, etc.
}

# Tracking delle azioni anziché solo i tasti
key_actions = {
    'accelerate': False,
    'brake': False,
    'steer_left': False,
    'steer_right': False
}

def setup_csv_logging():
    # Crea la directory se non esiste
    data_dir = "torcs_training_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Creata cartella {data_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(data_dir, f"telemetry_{timestamp}.csv")
    
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
def on_press(key):
    try:
        if hasattr(key, 'char'):
            # Registra l'azione basata sul tasto
            if key.char == 'w':
                key_actions['accelerate'] = True
            elif key.char == 's':
                key_actions['brake'] = True
            elif key.char == 'a':
                key_actions['steer_left'] = True
            elif key.char == 'd':
                key_actions['steer_right'] = True
            elif key.char == 'q':  
                controls['gear'] = max(-1, controls['gear'] - 1)
            elif key.char == 'e':  
                controls['gear'] = min(6, controls['gear'] + 1)
            elif key.char == 'r':  
                # Reset
                controls['throttle'] = 0.0
                controls['brake'] = 0.0
                controls['steer'] = 0.0
                key_actions['accelerate'] = False
                key_actions['brake'] = False
                key_actions['steer_left'] = False
                key_actions['steer_right'] = False
            
            # Applica le azioni ai controlli
            update_controls()
    except AttributeError:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char'):
            # Disattiva l'azione quando il tasto viene rilasciato
            if key.char == 'w':
                key_actions['accelerate'] = False
            elif key.char == 's':
                key_actions['brake'] = False
            elif key.char == 'a':
                key_actions['steer_left'] = False
            elif key.char == 'd':
                key_actions['steer_right'] = False
            
            # Aggiorna i controlli in base alle azioni attive
            update_controls()
    except AttributeError:
        pass
    
    if key == keyboard.Key.esc:
        return False

def update_controls():
    # Throttle e brake
    if key_actions['accelerate'] and not key_actions['brake']:
        controls['throttle'] += 0.2
        controls['brake'] = 0.0
    elif key_actions['brake'] and not key_actions['accelerate']:
        controls['throttle'] = 0.0
        controls['brake'] += 0.1
    elif not key_actions['accelerate'] and not key_actions['brake']:
        controls['throttle'] = 0.0
        controls['brake'] = 0.0
    
    # Sterzo
    if key_actions['steer_left'] and not key_actions['steer_right']:
        controls['steer'] += 0.2
    elif key_actions['steer_right'] and not key_actions['steer_left']:
        controls['steer'] -= 0.2
    elif not key_actions['steer_left'] and not key_actions['steer_right']:
        controls['steer'] = 0.0

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

# def drive_controller(telemetry):
#     """Controller di guida base (esempio)"""
#     steer = -telemetry['trackPos'] * 0.5  # Regolazione sterzo
#     throttle = 0.3 if abs(telemetry['trackPos']) < 0.8 else 0.1
#     brake = 0.0
#     gear = 3  # Cambio fisso
    
#     # Formato comandi per SCR
#     return f"(accel {throttle})(brake {brake})(gear {gear})(steer {steer})"
def drive_controller(telemetry):
    """Controller con input da tastiera"""
    # Format commands for SCR
    return f"(accel {controls['throttle']:.2f})(brake {controls['brake']:.2f})(gear {controls['gear']})(steer {controls['steer']:.2f})"

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
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

try:
    while True:
        # Ricezione dati telemetrici
        data, addr = sock.recvfrom(BUFFER_SIZE)
        
        telemetry = parse_telemetry(data.decode())
        print(telemetry)
        if not telemetry:
            continue
        log_telemetry(csv_filename, telemetry)
        
        # Calcolo azioni di guida
        command = drive_controller(telemetry)
        
        sock.sendto(command.encode(), (SERVER_IP, SERVER_PORT))
        
        # Debug
        # print(f"Pos: {telemetry['trackPos']:.2f} | Comando: {command}")

except KeyboardInterrupt:
    print("\nClient terminato")
finally:
    listener.stop()
    sock.close()