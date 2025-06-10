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
    'gear': 0  # Neutral = 0, First = 1, etc.
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

def list_available_models():
    """Lista tutti i modelli disponibili nella cartella models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("Cartella models non trovata!")
        return []
    
    model_folders = []
    for item in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, item)
        if os.path.isdir(folder_path):
            # Controlla se contiene file .keras
            keras_files = [f for f in os.listdir(folder_path) if f.endswith('.keras')]
            if keras_files:
                model_folders.append((item, keras_files))
    
    return model_folders

def select_model():
    """Permette di scegliere il modello da usare"""
    available_models = list_available_models()
    
    if not available_models:
        print("Nessun modello trovato nella cartella models/")
        return None, None
    
    print("Modelli disponibili:")
    for i, (folder_name, keras_files) in enumerate(available_models):
        print(f"{i+1}. {folder_name}")
        for keras_file in keras_files:
            print(f"   - {keras_file}")
    
    while True:
        try:
            choice = int(input("\nScegli il numero del modello: ")) - 1
            if 0 <= choice < len(available_models):
                folder_name, keras_files = available_models[choice]
                
                if len(keras_files) == 1:
                    selected_file = keras_files[0]
                else:
                    print("File .keras nella cartella:")
                    for i, keras_file in enumerate(keras_files):
                        print(f"{i+1}. {keras_file}")
                    
                    file_choice = int(input("Scegli il file: ")) - 1
                    if 0 <= file_choice < len(keras_files):
                        selected_file = keras_files[file_choice]
                    else:
                        print("Scelta non valida")
                        continue
                
                model_path = os.path.join("models", folder_name, selected_file)
                scaler_path = os.path.join("models", folder_name, "torcs_scaler.save")
                
                return model_path, scaler_path
            else:
                print("Scelta non valida")
        except ValueError:
            print("Inserisci un numero valido")

# Seleziona il modello
MODEL_PATH, SCALER_PATH = select_model()

if not MODEL_PATH:
    print("Nessun modello selezionato. Uscita...")
    exit()

print(f"Modello selezionato: {MODEL_PATH}")
print(f"Scaler: {SCALER_PATH}")

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

# Buffer per le sequenze temporali
SEQUENCE_LENGTH = 10
sequence_buffer = []

# Rimuovi il gear mapping - usa direttamente il valore predetto
# gear_mapping = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
# gear_pred = gear_mapping[int(prediction[0][3])]

# Il calcolo di gear_pred viene ora fatto dopo la predizione del modello nel ciclo principale.

# Feature order must match what the model was trained with
# Questo deve corrispondere alle 29 feature usate in training/data.py
sensor_features_for_model = [
    'angle', 
    'rpm', 
    'speedX', 
    'speedY', 
    'speedZ', 
    'trackPos'
]
sensor_features_for_model += [f'track_{i}' for i in range(19)] # track_0 to track_18
sensor_features_for_model += [f'wheelSpinVel_{i}' for i in range(4)] # wheelSpinVel_0 to wheelSpinVel_3

# Rinomina la variabile per chiarezza, dato che 'feature_order' era usata sotto
# feature_order = sensor_features_for_model # Vecchio nome, puoi mantenere questo se preferisci

try:
    while True:
        # Ricezione dati telemetrici
        data, addr = sock.recvfrom(BUFFER_SIZE)
        
        telemetry = parse_telemetry(data.decode())
        
        if not telemetry:
            continue
        
        # Prepare input data
        input_features = []
        for feature_name in sensor_features_for_model: 
            if feature_name in telemetry:
                input_features.append(telemetry[feature_name])
            elif feature_name.startswith('track_') and 'track' in telemetry:
                idx = int(feature_name.split('_')[1])
                if idx < len(telemetry['track']):
                    input_features.append(telemetry['track'][idx])
                else:
                    input_features.append(0.0)
            elif feature_name.startswith('wheelSpinVel_') and 'wheelSpinVel' in telemetry:
                idx = int(feature_name.split('_')[1])
                if idx < len(telemetry['wheelSpinVel']):
                    input_features.append(telemetry['wheelSpinVel'][idx])
                else:
                    input_features.append(0.0)
            else:
                print(f"Attenzione: feature '{feature_name}' non trovata nella telemetria, usando 0.0")
                input_features.append(0.0)
        
        if len(input_features) != 29:
            print(f"Errore: Numero di features ({len(input_features)}) non corrisponde a 29. Saltando.")
            continue
        
        # Aggiungi il frame corrente al buffer
        X = np.array(input_features)
        X_scaled = scaler.transform(X.reshape(1, -1))[0]  # Scala e ottieni il vettore 1D
        sequence_buffer.append(X_scaled)
        
        # Mantieni solo gli ultimi SEQUENCE_LENGTH frames
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)
        
        # Se non abbiamo abbastanza frames, usa controlli di default
        if len(sequence_buffer) < SEQUENCE_LENGTH:
            print(f"Raccogliendo dati... {len(sequence_buffer)}/{SEQUENCE_LENGTH}")
            # Controlli di default mentre costruiamo la sequenza
            command = f"(accel 0.3)(brake 0.0)(gear 1)(steer 0.0)"
            sock.sendto(command.encode(), addr)
            continue
        
        # Prepara input per il modello: (1, SEQUENCE_LENGTH, 29)
        model_input = np.array(sequence_buffer).reshape(1, SEQUENCE_LENGTH, 29)
        
        # Fai la predizione
        prediction = model.predict(model_input, verbose=0)
        
        # Il modello restituisce (1, 4) con [steer, throttle, brake, gear]
        steer_pred = prediction[0][0]
        throttle_pred = prediction[0][1] 
        brake_pred = prediction[0][2]
        gear_pred = int(round(prediction[0][3]))  # Usa direttamente il valore predetto
        gear_pred = max(-1, min(6, gear_pred))    # Assicurati che sia nel range corretto
        
        actions = {
            'steer': float(steer_pred),
            'throttle': float(throttle_pred),
            'brake': float(brake_pred),
            'gear': gear_pred
        }
        
        # Send command to TORCS
        command = f"(accel {actions['throttle']:.3f})(brake {actions['brake']:.3f})(gear {actions['gear']})(steer {actions['steer']:.3f})"
        sock.sendto(command.encode(), addr)
        
        # Print debug info
        print(f"Steer: {actions['steer']:.2f} | Accel: {actions['throttle']:.2f} | "
              f"Brake: {actions['brake']:.2f} | Gear: {actions['gear']}")

except KeyboardInterrupt:
    print("\nClient terminato")
finally:
    sock.close()