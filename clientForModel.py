
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

# Load model and scaler
MODEL_PATH = "models/best_torcs_model.h5"
SCALER_PATH = "scaler/torcs_scaler.save"

# Initialize the AI driver
from keras.models import load_model
import joblib
import numpy as np

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modello e scaler caricati con successo")
except Exception as e:
    print(f"Errore caricamento modello/scaler: {e}")
    sock.close()
    exit()

# Map gear classes to actual gear values
gear_mapping = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

# Feature order must match what the model was trained with
feature_order = [
    'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced',
    'fuel', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos', 'z',
    'lastLapTime'
]
feature_order += [f'track_{i}' for i in range(19)]  # 19 track sensors
feature_order += [f'wheelSpinVel_{i}' for i in range(4)]  # 4 wheel spin sensors
feature_order += [f'focus_{i}' for i in range(5)]  # 5 focus sensors

try:
    while True:
        # Ricezione dati telemetrici
        data, addr = sock.recvfrom(BUFFER_SIZE)
        
        telemetry = parse_telemetry(data.decode())
        
        if not telemetry:
            continue
        
        # Prepare input data
        input_features = []
        for feature in feature_order:
            if feature in telemetry:
                input_features.append(telemetry[feature])
            elif feature.startswith('track_') and 'track' in telemetry:
                # Extract track sensor index
                idx = int(feature.split('_')[1])
                if idx < len(telemetry['track']):
                    input_features.append(telemetry['track'][idx])
                else:
                    input_features.append(0.0)
            elif feature.startswith('wheelSpinVel_') and 'wheelSpinVel' in telemetry:
                idx = int(feature.split('_')[1])
                if idx < len(telemetry['wheelSpinVel']):
                    input_features.append(telemetry['wheelSpinVel'][idx])
                else:
                    input_features.append(0.0)
            elif feature.startswith('focus_') and 'focus' in telemetry:
                idx = int(feature.split('_')[1])
                if idx < len(telemetry['focus']):
                    input_features.append(telemetry['focus'][idx])
                else:
                    input_features.append(0.0)
            else:
                input_features.append(0.0)
                
        # Make prediction
        print('dio', input_features, 'dio')
        X = np.array(input_features).reshape(1, -1)
        
        X_scaled = scaler.transform(X)
        steer_pred, throttle_pred, brake_pred, gear_probs = model.predict(X_scaled, verbose=0)
        
        actions = {
            'steer': float(steer_pred[0][0]),
            'throttle': float(throttle_pred[0][0]),
            'brake': float(brake_pred[0][0]),
            'gear': gear_mapping[np.argmax(gear_probs)]
        }
        
        # Send command to TORCS
        command = f"(accel {actions['throttle']:.3f})(brake {actions['brake']:.3f})(gear {actions['gear']})(steer {actions['steer']:.3f})"
        sock.sendto(command.encode(), addr)
        
        # Print debug info
        print(f"Speed: {telemetry.get('speedX', 0)*3.6:.1f} km/h | "
              f"Steer: {actions['steer']:.2f} | Accel: {actions['throttle']:.2f} | "
              f"Brake: {actions['brake']:.2f} | Gear: {actions['gear']}")

except KeyboardInterrupt:
    print("\nClient terminato")
finally:
    sock.close()