import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_prepare_data(data_dir, destination_dir=""):
    """Carica e prepara i dati dai file CSV con struttura specificata"""

    all_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    list_dataframes = []
    
    if not all_files:
        raise ValueError(f"Nessun file CSV trovato nella directory: {data_dir}")

    print(f"Trovati {len(all_files)} file CSV in '{data_dir}'. Inizio caricamento...")

    for file in all_files:
        file_path = os.path.join(data_dir, file)
        try:
            dataframes = pd.read_csv(file_path)
            list_dataframes.append(dataframes)
            print(f"Caricato con successo: {file}")
        except Exception as e:
            print(f"Errore caricamento {file}: {e}")
    
    if not list_dataframes:
        # Questa condizione potrebbe essere ridondante se la precedente `if not all_files:` cattura già il caso,
        # ma la manteniamo per sicurezza nel caso alcuni file esistano ma nessuno sia leggibile.
        raise ValueError("Nessun dato CSV valido è stato caricato dalla directory specificata.")
        
    full_dataframes = pd.concat(list_dataframes, ignore_index=True)
    print(f"Dati concatenati. Shape totale: {full_dataframes.shape}")

    sensor_features = [
        'angle', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos', 
        'track_0','track_1','track_2','track_3','track_4','track_5','track_6','track_7','track_8',
        'track_9','track_10','track_11','track_12','track_13','track_14','track_15','track_16',
        'track_17','track_18',
        'wheelSpinVel_0', 'wheelSpinVel_1', 'wheelSpinVel_2', 'wheelSpinVel_3'
    ]
    targets = ['steer', 'throttle', 'brake', 'gear']

    # Verifica la presenza di tutte le colonne necessarie
    missing_sensor_features = [col for col in sensor_features if col not in full_dataframes.columns]
    missing_targets = [col for col in targets if col not in full_dataframes.columns]

    if missing_sensor_features:
        raise ValueError(f"Colonne mancanti per le features sensoriali: {missing_sensor_features}")
    if missing_targets:
        raise ValueError(f"Colonne mancanti per i targets: {missing_targets}")

    sensorsData_df = full_dataframes[sensor_features]
    
    # Gestione dei valori NaN prima della normalizzazione
    if sensorsData_df.isnull().values.any():
        print(f"Attenzione: Trovati NaN nelle features sensoriali. Verranno riempiti con la media della colonna.")
        # Opzione: riempire con la media o mediana. Per semplicità, usiamo la media.
        # È consigliabile investigare la causa dei NaN.
        for col in sensorsData_df.columns[sensorsData_df.isnull().any()]:
            sensorsData_df[col] = sensorsData_df[col].fillna(sensorsData_df[col].mean())


    sensorsData = sensorsData_df.values
    scaler = StandardScaler()
    normalized_SensorsData = scaler.fit_transform(sensorsData)
    
    targetsData = full_dataframes[targets].copy()

    targetsData['steer'] = targetsData['steer'].clip(-1.0, 1.0)
    targetsData['throttle'] = targetsData['throttle'].clip(0.0, 1.0)
    targetsData['brake'] = targetsData['brake'].clip(0.0, 1.0)
    
    # Rimuovi completamente il gear mapping - lascia i valori originali
    # gear_mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
    # targetsData.loc[:, 'gear'] = targetsData['gear'].map(gear_mapping)

    # Invece, assicurati solo che le marce siano nel range corretto
    targetsData['gear'] = targetsData['gear'].clip(-1, 6).astype(int)

    if targetsData['gear'].isnull().any():
        num_nan_gears = targetsData['gear'].isnull().sum()
        print(f"Attenzione: Trovati {num_nan_gears} valori di 'gear' non mappabili (NaN dopo la mappatura).")
        
        # Gestione dei NaN in 'gear' (rimozione delle righe corrispondenti)
        # Trova gli indici delle righe con NaN in 'gear'
        nan_gear_indices = targetsData[targetsData['gear'].isnull()].index
        
        # Rimuovi queste righe da targetsData
        targetsData = targetsData.drop(nan_gear_indices)
        
        # Rimuovi le stesse righe da normalized_SensorsData (che è un array NumPy)
        normalized_SensorsData = np.delete(normalized_SensorsData, nan_gear_indices, axis=0)
        
        print(f"Rimosse {len(nan_gear_indices)} righe a causa di valori 'gear' non mappabili.")
        if normalized_SensorsData.shape[0] == 0:
            raise ValueError("Nessun dato rimasto dopo la rimozione delle righe con 'gear' non mappabili.")


    targetsData['gear'] = targetsData['gear'].astype(int)

    # Salva lo scaler per uso futuro
    scaler_filename = 'torcs_scaler.save'
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)
        scaler_path = os.path.join(destination_dir, scaler_filename)
    else:
        # Salva nella directory corrente se destination_dir non è specificato
        scaler_path = scaler_filename
    
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvato in: {scaler_path}")
    
    return normalized_SensorsData, targetsData

    
if __name__ == "__main__":
    # Definisci la directory contenente i dati di training CSV
    # Assicurati che questa directory esista e contenga i tuoi file.
    data_directory = "torcs_training_data" 
    
    # Definisci la directory dove salvare gli artefatti (es. lo scaler)
    # Se lasciata vuota, lo scaler verrà salvato nella directory corrente.
    artifacts_output_dir = "training_artifacts"

    # Crea la directory dei dati se non esiste (per test iniziali, ma dovrebbe contenere dati)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Directory '{data_directory}' creata. Assicurati di popolarla con i file CSV.")
        # Non si esce, ma load_and_prepare_data solleverà un errore se è vuota.

    try:
        print(f"Avvio pre-elaborazione dati da: '{data_directory}'")
        X_train, Y_train = load_and_prepare_data(data_dir=data_directory, destination_dir=artifacts_output_dir)
        
        print("\nPre-elaborazione completata.")
        print(f"Forma di X_train (features normalizzate): {X_train.shape}")
        print(f"Forma di Y_train (targets processati): {Y_train.shape}")
        
        # Esempio di visualizzazione delle prime righe (opzionale)
        if X_train.shape[0] > 0:
            print("\nPrime 3 righe di X_train:")
            print(X_train[:3])
            print("\nPrime 3 righe di Y_train:")
            print(Y_train.head(3))

    except ValueError as e:
        print(f"\nErrore durante la pre-elaborazione dei dati: {e}")
    except FileNotFoundError as e:
        print(f"\nErrore: Directory o file non trovato: {e}")
    except Exception as e:
        print(f"\nErrore imprevisto durante la pre-elaborazione: {e}")
