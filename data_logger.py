import os
import csv
import socket
from datetime import datetime

FIELDNAMES = [
    'timestamp', 'angle', 'curLapTime', 'damage', 'distFromStart', 
    'distRaced', 'fuel', 'gear', 'rpm', 'speedX', 'speedY', 'speedZ', 
    'trackPos', 'throttle', 'brake', 'steer', 'lastLapTime', 'z'
]
FIELDNAMES.extend([f'track_{i}' for i in range(19)])
FIELDNAMES.extend([f'wheelSpinVel_{i}' for i in range(4)])
FIELDNAMES.extend([f'focus_{i}' for i in range(5)])

def setup_csv_logging():
    """
    Crea la cartella per il logging e il file CSV con le intestazioni.
    Restituisce il percorso del file CSV.
    """
    data_dir = "torcs_training_data"
    if not os.path.exists(data_dir):

        os.makedirs(data_dir)
        print(f"Creata cartella {data_dir}")

    pc_name = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(data_dir, f"{pc_name}_telemetry_{timestamp}.csv")


    with open(csv_filename, 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
    
    return csv_filename

def log_telemetry(csv_filename, telemetry, controls):
    """
    Registra i dati telemetrici e di controllo in un file CSV.
    """
    with open(csv_filename, 'a', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction='ignore')
        
        row = {'timestamp': datetime.now().isoformat()}
        
        for key, value in telemetry.items():
            if key in FIELDNAMES:
                row[key] = value
        
        for key in ['track', 'wheelSpinVel', 'focus']:
            if key in telemetry:
                for i, val in enumerate(telemetry[key]):
                    row[f'{key}_{i}'] = val
        
        row.update(controls)
        
        writer.writerow(row)
