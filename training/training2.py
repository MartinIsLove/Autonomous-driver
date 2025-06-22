import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed  # Aggiunto
import winsound
import time
from sklearn.preprocessing import LabelEncoder
# ===============================
# CONFIGURAZIONE GLOBALE
# ===============================

# Feature sets per ogni controllo
FEATURE_SETS = {
    'steer': [
        'angle', 'trackPos', 'speedX', 'speedY',
        'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
        'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
        'track_10', 'track_11', 'track_12', 'track_13', 'track_14',
        'track_15', 'track_16', 'track_17', 'track_18'
    ],
    'throttle': [
        'speedX', 'speedY', 
        'angle', 'trackPos','brake',
        'rpm',
        'track_7', 'track_8', 'track_9','track_10', 'track_11'
    ],
    'brake': [
        'speedX', 'speedY',
        'angle', 'trackPos',
        'track_7', 'track_8', 'track_9','track_10', 'track_11'
    ],
    'gear': [
        'speedX', 'trackPos', 'rpm', 
        'angle', 'track_7', 'track_8', 'track_9', 'track_10', 'track_11',
        'throttle', 'brake', 'speedY'
    ]
}

# Modelli ottimali identificati
OPTIMAL_MODELS = {
    'steer': 'nn',      # Neural Network (R²: 0.8417)
    'throttle': 'nn',   # Neural Network (R²: 0.7748) 
    'brake': 'nn',      # Random Forest (R²: 0.4260)
    'gear': 'rf'        # Random Forest (Accuracy: 0.9780)
}

# ===============================
# FUNZIONI CORE
# ===============================

def load_data(data_dir):
    """Carica e combina tutti i file CSV di training"""
    all_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    list_dataframes = []
    
    if not all_files:
        raise ValueError(f"Nessun file CSV trovato in: {data_dir}")
    
    print(f"Caricamento {len(all_files)} file CSV...")
    
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
    """Training di un singolo modello"""
    print(f"\nTraining {target_name.upper()}...")
    
    X = dataset[FEATURE_SETS[target_name]]
    y = dataset[target_name]
    
    # Pulizia dati
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X, y = X[mask], y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = None
    
    if target_name in ['steer', 'throttle', 'brake']:  # Aggiunto 'brake'
        # Neural Network
        le = None
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Configurazioni specifiche per ciascun controllo
        
        if target_name == 'steer':
            model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                max_iter=2000, early_stopping=True, validation_fraction=0.1,
                random_state=42
            )
        elif target_name == 'throttle':
            model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                max_iter=2000, early_stopping=True, validation_fraction=0.1,
                random_state=42
            )
        elif target_name == 'brake':
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16), activation='relu',
                max_iter=2000, early_stopping=True, validation_fraction=0.1,
                random_state=42, alpha=0.01 
            )
        else:
            raise ValueError(f"Unknown target_name: {target_name}")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_test_enc = le.transform(y_test) if le else y_test
        score = r2_score(y_test_enc, y_pred)
        metric = "R²"
        
    else:  # gear (solo gear rimane Random Forest)
        # Random Forest Classifier
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=100,
            min_samples_split=20,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        metric = "Accuracy"
    
    print(f"   {target_name}: {metric} = {score:.4f}")
    return target_name, model, scaler, le

def train_models(data_dir):
    """Training parallelo di tutti i modelli"""
    dataset = load_data(data_dir)
    
    models = {}
    scalers = {}
    encoders = {}
    
    print("\n" + "="*50)
    print("TRAINING MODELLI PARALLELO")
    print("="*50)

    start_time = time.time()
    
    # Training parallelo
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(train_single_model, target, dataset): target
            for target in ['steer', 'throttle', 'brake', 'gear']
        }
        
        for future in as_completed(futures):
            target_name, model, scaler, encoder = future.result()
            models[target_name] = model
            if scaler:
                scalers[target_name] = scaler
            if encoder:
                encoders[target_name] = encoder
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTutti i modelli completati! Tempo totale: {elapsed_time:.2f} secondi")
    return models, scalers, encoders

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
    pred_brake = models['brake'].predict(X_input_brake)[0]
    pred_brake = np.clip(pred_brake, 0.0, 1.0)
    predictions['brake'] = pred_brake
    
    # ===============================
    # PREDIZIONE GEAR
    # ===============================
    features_gear = FEATURE_SETS['gear']
    X_input_gear = np.array([[sensor_data[f] for f in features_gear]])
    pred_gear = models['gear'].predict(X_input_gear)[0]
    pred_gear_enc = models['gear'].predict(X_input_gear)[0]
    pred_gear = encoders['gear'].inverse_transform([pred_gear_enc])[0]  # Decodifica l'etichetta
    # pred_gear = int(max(-1, min(6, pred_gear)))
    predictions['gear'] = pred_gear
    
    return predictions

def save_model(models, scalers, filepath):
    """Salva il modello trainato"""
    model_data = {
        'models': models,
        'scalers': scalers,
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
    
    print(f"Modello caricato da: {filepath}")
    return models, scalers

# ===============================
# TRAINING E TESTING
# ===============================

if __name__ == "__main__":
    # Training
    data_dir = './torcs_training_data'
    models, scalers, encoders = train_models(data_dir)
    
    # Salva il modello in una cartella con timestamp
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = f'models/model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'torcs_optimal_model.pkl')
    save_model(models, scalers, model_path)
    
    # Test con dati di esempio
    print("\n" + "="*50)
    print("TEST DEL MODELLO")
    print("="*50)
    
    # Carica dati per test
    test_data = load_data(data_dir)
    sample_idx = 1000
    
    # Prepara input di test
    test_input = {}
    all_features = set()
    for features in FEATURE_SETS.values():
        all_features.update(features)
    
    for feature in all_features:
        test_input[feature] = test_data[feature].iloc[sample_idx]
    
    # Predizione
    controls = predict(test_input, models, scalers)
    
    print("🔍 Confronto Predizione vs Realtà:")
    print("-" * 40)
    
    # STEER
    real_steer = test_data['steer'].iloc[sample_idx]
    pred_steer = controls['steer']
    diff_steer = abs(pred_steer - real_steer)
    print(f"steer    | Pred: {pred_steer:6.3f} | Real: {real_steer:6.3f} | Diff: {diff_steer:6.3f}")
    
    # THROTTLE
    real_throttle = test_data['throttle'].iloc[sample_idx]
    pred_throttle = controls['throttle']
    diff_throttle = abs(pred_throttle - real_throttle)
    print(f"throttle | Pred: {pred_throttle:6.3f} | Real: {real_throttle:6.3f} | Diff: {diff_throttle:6.3f}")
    
    # BRAKE
    real_brake = test_data['brake'].iloc[sample_idx]
    pred_brake = controls['brake']
    diff_brake = abs(pred_brake - real_brake)
    print(f"brake    | Pred: {pred_brake:6.3f} | Real: {real_brake:6.3f} | Diff: {diff_brake:6.3f}")
    
    # GEAR
    real_gear = test_data['gear'].iloc[sample_idx]
    pred_gear = controls['gear']
    diff_gear = abs(pred_gear - real_gear)
    print(f"gear     | Pred: {pred_gear:6.3f} | Real: {real_gear:6.3f} | Diff: {diff_gear:6.3f}")
    
    print(f"\nModello ottimale pronto per l'uso!")
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
