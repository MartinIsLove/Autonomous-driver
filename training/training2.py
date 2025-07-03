import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import winsound
# Per grafici
import matplotlib.pyplot as plt

# ===============================
# CONFIGURAZIONE GLOBALE
# ===============================

# Feature sets per ogni controllo
FEATURE_SETS = {
    'steer': [
        'angle', 'trackPos', 
        'speedX', 'speedY',
        'brake',
        'throttle', 
        
        'track_0',
        'track_1', 'track_2',
        'track_3', 'track_4',
        'track_5', 'track_6',
        'track_7', 'track_8',
        'track_9', 'track_10',
        'track_11', 'track_12',
        'track_13', 'track_14',
        'track_15', 'track_16',
        'track_17', 'track_18'
    ],
    'throttle': [
        'brake', 'steer',
        'speedX', 'speedY',
        'angle', 'trackPos',
        'rpm', 'gear',
        
        'track_9',
        'track_8', 'track_10',
        'track_6', 'track_12',
        'track_4', 'track_14'
    ],

    'brake': [
        'throttle', 'steer',
        'speedX', 'speedY',
        'angle', 'trackPos',
        'rpm',
        'gear',
        'track_0', 'track_1',
        'track_2', 'track_3',
        'track_4', 'track_5',
        'track_6', 'track_7',
        'track_8', 'track_9',
        'track_10', 'track_11',
        'track_12', 'track_13',
        'track_14', 'track_15',
        'track_16', 'track_17',
        'track_18'
    ],

    'gear': [
        'rpm',              
        'speedX',           
        'speedY',           
        'angle',            
        'trackPos',         
        'throttle',         
        'brake',            
        'steer',

        'track_9',
        'track_8', 'track_10',
        'track_6', 'track_12',
        'track_4', 'track_14'        
    ]
}


OPTIMAL_MODELS = {
    'steer': 'nn',
    'throttle': 'nn',
    'brake': 'nn',
    'gear': 'rf'
}

# ===============================
# FUNZIONI 
# ===============================

def load_data(data_dir):
    """Carica e combina tutti i file CSV di training"""
    all_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    list_dataframes = []
    
    if not all_files:
        raise ValueError(f"Nessun file CSV trovato in: {data_dir}")
    
    print(f"Caricamento {len(all_files)} file CSV..")
    
    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            list_dataframes.append(df)
            print(f"{file}")
        except Exception as e:
            print(f"Errore {file}: {e}")
    
    if not list_dataframes:
        raise ValueError("Nessun dato valido caricato")
    
    dataset = pd.concat(list_dataframes, ignore_index=True)
    print(f"Dataset finale: {dataset.shape[0]} righe, {dataset.shape[1]} colonne")
    
    return dataset

def train_single_model(target_name, dataset):
    print(f"\nTraining {target_name.upper()}..")
    X = dataset[FEATURE_SETS[target_name]]
    y = dataset[target_name]
    
    # Pulizia dati
    # mask = ~(X.isnull().any(axis=1) | y.isnull())
    # X, y = X[mask], y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = None
    encoder = None
    history = {}
    if target_name in ['steer', 'throttle', 'brake']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = None
        if target_name == 'steer':
            model = MLPRegressor(
                hidden_layer_sizes=(64, 128, 192, 128, 64, 32),
                activation='tanh',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                verbose=False
            )
        elif target_name == 'throttle':
            model = MLPRegressor(
                hidden_layer_sizes=(400, 300, 200, 100, 50),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                verbose=False
            )
        elif target_name == 'brake':
            model = MLPRegressor(
                hidden_layer_sizes=(120, 100, 80, 60, 40, 20),
                activation='tanh',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                verbose=False
            )
        if model is not None:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            metric = "R²"

            if hasattr(model, 'loss_curve_'):
                history['loss_curve'] = model.loss_curve_
            else:
                history['loss_curve'] = []
            history['r2'] = score
        else:
            score = 0
            metric = "R²"
            history['loss_curve'] = []
            history['r2'] = score
    else:
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        
        model = RandomForestClassifier(
            n_estimators=128,
            max_depth=16,
            min_samples_split=20,
            min_samples_leaf=6,
            max_features='log2',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_encoded)
        y_pred_encoded = model.predict(X_test)
        score = accuracy_score(y_test_encoded, y_pred_encoded)
        metric = "Accuracy"
        history['accuracy'] = score
    
    print(f"   {target_name}: {metric} = {score:.4f}")
    return target_name, model, scaler, encoder, history

def train_models(data_dir):
    dataset = load_data(data_dir)
    
    models = {}
    scalers = {}
    encoders = {}
    histories = {}
    
    print("\n" + "="*50)
    print("TRAINING MODELLI PARALLELO")
    print("="*50)
    
    # Training parallelo
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(train_single_model, target, dataset): target
            for target in ['steer', 'throttle', 'brake', 'gear']
        }
        
        for future in as_completed(futures):
            target_name, model, scaler, encoder, history = future.result()
            models[target_name] = model
            if scaler:
                scalers[target_name] = scaler
            if encoder:
                encoders[target_name] = encoder
            histories[target_name] = history
    
    print(f"\nTutti i modelli completati!")
    return models, scalers, encoders, histories

def predict(sensor_data, models, scalers, encoders):
    """
    Predice i controlli completi per un dato stato dei sensori
    
    Args:
        sensor_data (dict): Dati sensori TORCS
        models (dict): Modelli trainati
        scalers (dict): Scalers per neural networks
        
    Returns:
        dict: Controlli predetti (steer, throttle, brake, gear)
    """
    
    predictions = {}
    
    # ===============================
    # PREDIZIONE STEER
    # ===============================
    features_steer = FEATURE_SETS['steer']
    X_input_steer = np.array([[sensor_data[f] for f in features_steer]])
    X_input_steer = scalers['steer'].transform(X_input_steer)
    pred_steer = models['steer'].predict(X_input_steer)[0]
    pred_steer = np.clip(pred_steer, -1.0, 1.0)
    predictions['steer'] = pred_steer
    
    # ===============================
    # PREDIZIONE THROTTLE
    # ===============================
    features_throttle = FEATURE_SETS['throttle']
    X_input_throttle = np.array([[sensor_data[f] for f in features_throttle]])
    X_input_throttle = scalers['throttle'].transform(X_input_throttle)
    pred_throttle = models['throttle'].predict(X_input_throttle)[0]
    pred_throttle = np.clip(pred_throttle, 0.0, 1.0)
    predictions['throttle'] = pred_throttle
    
    # ===============================
    # PREDIZIONE BRAKE
    # ===============================
    features_brake = FEATURE_SETS['brake']
    X_input_brake = np.array([[sensor_data[f] for f in features_brake]])
    X_input_brake = scalers['brake'].transform(X_input_brake)
    pred_brake = models['brake'].predict(X_input_brake)[0]
    pred_brake = np.clip(pred_brake, 0.0, 1.0)
    predictions['brake'] = pred_brake
    
    # ===============================
    # PREDIZIONE GEAR
    # ===============================
    features_gear = FEATURE_SETS['gear']
    X_input_gear = np.array([[sensor_data[f] for f in features_gear]])
    pred_gear_encoded = models['gear'].predict(X_input_gear)[0]
    pred_gear = encoders['gear'].inverse_transform([pred_gear_encoded])[0]
    predictions['gear'] = pred_gear
    
    return predictions

def save_model(models, scalers, encoders, filepath):
    """Salva il modello trainato"""
    model_data = {
        'models': models,
        'scalers': scalers,
        'encoders': encoders,
        'feature_sets': FEATURE_SETS,
        'optimal_models': OPTIMAL_MODELS
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Modello salvato in: {filepath}")

def load_model(filepath):
    """Carica un modello salvato"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    models = model_data['models']
    scalers = model_data['scalers']
    encoders = model_data.get('encoders', {})
    
    print(f"Modello caricato da: {filepath}")
    return models, scalers, encoders

# ===============================
# TRAINING E TESTING
# ===============================

if __name__ == "__main__":

    import time
    start = time.time()

    data_dir = './torcs_training_data'
    models, scalers, encoders, histories = train_models(data_dir)
    
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = f'models/model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'torcs_optimal_model.pkl')
    save_model(models, scalers, encoders, model_path)

    
    print(f"\nModello pronto per l'uso!")
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
    
    # ===============================
    # GRAFICI METRICHE
    # ===============================
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Metriche Training Modelli')
    # STEER
    steer_hist = histories.get('steer', {})
    axs[0, 0].plot(steer_hist.get('loss_curve', []), label='Loss')
    axs[0, 0].set_title(f"Steer - R²: {steer_hist.get('r2', 0):.3f}")
    axs[0, 0].set_xlabel('Epoca')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    # THROTTLE
    throttle_hist = histories.get('throttle', {})
    axs[0, 1].plot(throttle_hist.get('loss_curve', []), label='Loss')
    axs[0, 1].set_title(f"Throttle - R²: {throttle_hist.get('r2', 0):.3f}")
    axs[0, 1].set_xlabel('Epoca')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    # BRAKE
    brake_hist = histories.get('brake', {})
    axs[1, 0].plot(brake_hist.get('loss_curve', []), label='Loss')
    axs[1, 0].set_title(f"Brake - R²: {brake_hist.get('r2', 0):.3f}")
    axs[1, 0].set_xlabel('Epoca')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    # GEAR
    gear_hist = histories.get('gear', {})
    axs[1, 1].bar(['Accuracy'], [gear_hist.get('accuracy', 0)])
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title(f"Gear - Accuracy: {gear_hist.get('accuracy', 0):.3f}")
    axs[1, 1].set_ylabel('Accuracy')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

    end = time.time()
    print(f"\nTempo totale di esecuzione: {end - start:.2f} secondi")
