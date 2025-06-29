import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import winsound

# ===============================
# CONFIGURAZIONE GLOBALE
# ===============================

# Feature sets per ogni controllo
FEATURE_SETS = {
    'steer': [
        'angle', 'trackPos', 
        'speedX', 'speedY',
        'brake',
        'throttle',         # Aggiunta - accelerazione influenza sterzo
        
        # Sensori pista ottimizzati per steering
        'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
        'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
        'track_10', 'track_11', 'track_12', 'track_13', 'track_14',
        'track_15', 'track_16', 'track_17', 'track_18',
    ],
    'throttle': [
        'brake', 'steer',
        'speedX', 'speedY',
        'angle', 'trackPos',
        'rpm', 'gear',
        
        # Solo sensori essenziali
        'track_9',                    # Frontale centrale
        'track_8', 'track_10',        # Frontali laterali
        'track_6', 'track_12',        # Per curve
        'track_4', 'track_14',        # Per curve ampie
    ],

    'brake': [
        'throttle', 'steer',
        'speedX', 'speedY',
        'angle', 'trackPos',
        'rpm',              # Per engine brake
        'gear',             # Marcia influenza frenata
        
        # Sensori pista ottimizzati (solo i più rilevanti)
        'track_8', 'track_9', 'track_10',  # Frontali
        'track_7', 'track_11',             # Laterali vicini
        'track_6', 'track_12',             # Laterali medi
        'track_5', 'track_13',             # Per curve ampie
        'track_0', 'track_18',             # Laterali estremi
    ],

    'gear': [
        'rpm',              # Fondamentale - giri motore
        'speedX',           # Velocità longitudinale
        'speedY',           # Velocità laterale (per curve)
        'angle',            # Angolo auto rispetto alla pista
        'trackPos',         # Posizione sulla pista
        'throttle',         # Accelerazione corrente
        'brake',            # Frenata corrente
        'steer',            # Sterzo (per curve strette)
        
        # Sensori pista (per anticipare curve e rettilinei)
        'track_9',          # Sensore frontale centrale
        'track_8', 'track_10',  # Sensori frontali laterali
        'track_7', 'track_11',  # Sensori più laterali
        'track_6', 'track_12',  # Per curve più ampie
    ]
}


OPTIMAL_MODELS = {
    'steer': 'nn',
    'throttle': 'nn',
    'brake': 'nn',
    'gear': 'nn'
}

# ===============================
# FUNZIONI CORE
# ===============================

def filter_bad_data(dataset):
    """
    Filter out bad driving data (crashes, excessive speed, lane violations)
    """
    print("Filtering bad driving data...")
    initial_size = len(dataset)
    
    # Remove data where car went off track (trackPos should be reasonable)
    dataset = dataset[(dataset['trackPos'] >= -1.0) & (dataset['trackPos'] <= 1.0)]
    
    # # Remove data with excessive speed (adjust threshold based on your data)
    # speed_threshold = np.percentile(dataset['speedX'], 95)  # Remove top 5% fastest
    # dataset = dataset[dataset['speedX'] <= speed_threshold]
    
    # Remove data where car is going backwards
    dataset = dataset[dataset['speedX'] >= 0]
    
    # Remove data with extreme steering angles that might indicate crashes
    # dataset = dataset[(dataset['steer'] >= -1.0) & (dataset['steer'] <= 1.0)]
    
    # Remove data with simultaneous high throttle and brake (conflicting actions)
    dataset = dataset[~((dataset['throttle'] > 0.5) & (dataset['brake'] > 0.5))]
    
    # Remove data with very low track sensor readings (too close to walls)
    track_cols = [f'track_{i}' for i in range(19)]
    min_distance = 2.0  # Minimum safe distance to track edges
    for col in track_cols:
        if col in dataset.columns:
            dataset = dataset[dataset[col] >= min_distance]
    
    filtered_size = len(dataset)
    removed_count = initial_size - filtered_size
    removal_percentage = (removed_count / initial_size) * 100
    
    print(f"Data filtering complete:")
    print(f"  - Initial size: {initial_size:,} samples")
    print(f"  - Filtered size: {filtered_size:,} samples")
    print(f"  - Removed: {removed_count:,} samples ({removal_percentage:.1f}%)")
    
    return dataset

def balance_steering_data(dataset, steer_range=0.2, remap_factor=5):
    """
    Balance steering data by limiting range, ensuring good distribution, and remapping for better learning
    """
    print(f"Balancing steering data to range ±{steer_range} with {remap_factor}x remapping...")
    
    # Show original distribution
    original_steer = dataset['steer']
    print(f"Original steering range: [{original_steer.min():.3f}, {original_steer.max():.3f}]")
    
    # Clip steering to desired range
    dataset = dataset.copy()
    dataset['steer'] = np.clip(dataset['steer'], -steer_range, steer_range)
    
    # CRUCIAL: Remap steering angles by factor of 5 for better neural network learning
    dataset['steer'] = dataset['steer'] * remap_factor
    print(f"Steering angles remapped by factor of {remap_factor}")
    print(f"New steering range: [{dataset['steer'].min():.3f}, {dataset['steer'].max():.3f}]")
    
    # Balance the dataset by removing over-represented steering angles
    # Create bins for steering angles (now in remapped range)
    remapped_range = steer_range * remap_factor
    bins = np.linspace(-remapped_range, remapped_range, 21)  # 20 bins
    dataset['steer_bin'] = pd.cut(dataset['steer'], bins, labels=False)
    
    # Find minimum count across bins to balance
    bin_counts = dataset['steer_bin'].value_counts()
    min_count = int(bin_counts.median())  # Use median instead of min to avoid too aggressive filtering
    
    # Sample equally from each bin
    balanced_dfs = []
    for bin_idx in range(len(bins)-1):
        bin_data = dataset[dataset['steer_bin'] == bin_idx]
        if len(bin_data) > 0:
            sample_size = min(len(bin_data), min_count)
            balanced_dfs.append(bin_data.sample(n=sample_size, random_state=42))
    
    balanced_dataset = pd.concat(balanced_dfs, ignore_index=True)
    balanced_dataset = balanced_dataset.drop('steer_bin', axis=1)
    
    print(f"Steering data balanced:")
    print(f"  - Original size: {len(dataset):,} samples")
    print(f"  - Balanced size: {len(balanced_dataset):,} samples")
    print(f"  - Reduction: {((len(dataset) - len(balanced_dataset)) / len(dataset) * 100):.1f}%")
    
    return balanced_dataset

def analyze_gear_distribution(dataset):
    """Analizza la distribuzione delle marce nel dataset"""
    gear_counts = dataset['gear'].value_counts().sort_index()
    print("\n📊 Distribuzione marce nel dataset:")
    print("-" * 40)
    total_samples = len(dataset)
    
    for gear, count in gear_counts.items():
        percentage = (count / total_samples) * 100
        bar = "█" * int(percentage / 2)  # Visual bar
        print(f"  Gear {gear}: {count:>8,} samples ({percentage:>5.1f}%) {bar}")
    
    # Calcola bilanciamento
    max_count = gear_counts.max()
    min_count = gear_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n  Ratio sbilanciamento: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("  ⚠️  ATTENZIONE: Dataset molto sbilanciato!")
    elif imbalance_ratio > 5:
        print("  ⚠️  Dataset moderatamente sbilanciato")
    else:
        print("  ✅ Dataset relativamente bilanciato")
    
    return gear_counts

def balance_gear_data(dataset, target_samples_per_gear=None):
    """Balance gear data per migliorare la predizione"""
    print("\  Bilanciamento dati gear...")
    
    # Analizza distribuzione attuale
    gear_counts = dataset['gear'].value_counts().sort_index()
    
    # Determina il target samples per gear
    if target_samples_per_gear is None:
        # Usa la mediana come target (compromise tra min e max)
        target_samples_per_gear = max(2000, int(gear_counts.median()))
    
    print(f"Target samples per gear: {target_samples_per_gear:,}")
    
    # Bilancia i dati
    balanced_dfs = []
    resampling_info = []
    
    for gear in sorted(dataset['gear'].unique()):
        gear_data = dataset[dataset['gear'] == gear]
        current_count = len(gear_data)
        
        if current_count > target_samples_per_gear:
            # Sottocampiona (undersampling)
            sampled_data = gear_data.sample(n=target_samples_per_gear, random_state=42)
            balanced_dfs.append(sampled_data)
            resampling_info.append(f"  Gear {gear}: {current_count:>6,} → {target_samples_per_gear:>6,} (undersampled)")
            
        elif current_count < target_samples_per_gear:
            # Sovracampiona con SMOTE-like approach (duplicazione intelligente)
            n_duplicates = target_samples_per_gear // current_count
            remainder = target_samples_per_gear % current_count
            
            # Duplica i dati esistenti
            duplicated_data = pd.concat([gear_data] * n_duplicates, ignore_index=True)
            
            # Aggiungi il remainder campionando casualmente
            if remainder > 0:
                extra_data = gear_data.sample(n=remainder, random_state=42)
                duplicated_data = pd.concat([duplicated_data, extra_data], ignore_index=True)
            
            resampling_info.append(f"  Gear {gear}: {current_count:>6,} → {len(duplicated_data):>6,} (oversampled)")
            
        else:
            # Già bilanciato
            balanced_dfs.append(gear_data)
            resampling_info.append(f"  Gear {gear}: {current_count:>6,} → {current_count:>6,} (unchanged)")
    
    # Combina tutti i dati bilanciati
    balanced_dataset = pd.concat(balanced_dfs, ignore_index=True)
    
    print("Risultati bilanciamento:")
    for info in resampling_info:
        print(info)
    
    print(f"\nDataset gear bilanciato:")
    print(f"  - Dimensione originale: {len(dataset):,} samples")
    print(f"  - Dimensione bilanciata: {len(balanced_dataset):,} samples")
    
    return balanced_dataset

def load_data(data_dir):
    """Carica e combina tutti i file CSV di training con filtering"""
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
    print(f"Dataset iniziale: {dataset.shape[0]} righe, {dataset.shape[1]} colonne")
    
    # Apply data filtering
    dataset = filter_bad_data(dataset)
    
    # NUOVO: Analizza distribuzione gear PRIMA del bilanciamento
    print("\n" + "="*50)
    print("ANALISI DISTRIBUZIONE GEAR - PRIMA")
    print("="*50)
    analyze_gear_distribution(dataset)
    
    # Balance steering data
    dataset = balance_steering_data(dataset, steer_range=0.2)
    
    # NUOVO: Balance gear data
    print("\n" + "="*50)
    print("BILANCIAMENTO DATI GEAR")
    print("="*50)
    dataset = balance_gear_data(dataset, target_samples_per_gear=3000)
    
    # NUOVO: Analizza distribuzione gear DOPO il bilanciamento
    print("\n" + "="*50)
    print("ANALISI DISTRIBUZIONE GEAR - DOPO")
    print("="*50)
    analyze_gear_distribution(dataset)
    
    print(f"Dataset finale processato: {dataset.shape[0]} righe, {dataset.shape[1]} colonne")
    
    return dataset

def train_single_model(target_name, dataset):
    """Training di un singolo modello con remapping ottimizzato"""
    print(f"\nTraining {target_name.upper()}...")
    
    X = dataset[FEATURE_SETS[target_name]]
    y = dataset[target_name]
    
    # Pulizia dati
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X, y = X[mask], y[mask]
    
    # Split with reduced test size for more training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if target_name == 'gear' else None
    )
    
    scaler = None
    encoder = None
    
    if target_name in ['steer', 'throttle', 'brake']:
        # Neural Network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if target_name == 'steer':
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), 
                activation='tanh',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42
            )
        elif target_name == 'throttle':
            model = MLPRegressor(
                hidden_layer_sizes=(128, 128, 64, 32), 
                activation='tanh',
                max_iter=2000, 
                early_stopping=True, 
                validation_fraction=0.15,
                random_state=42
            )
        elif target_name == 'brake':
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                max_iter=2000,
                early_stopping=True, 
                validation_fraction=0.15,
                random_state=42
            )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = r2_score(y_test, y_pred)
        metric = "R²"
        
    else:
        # Neural Network Classifier per gear - MIGLIORATO
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        
        # Modello gear più potente
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),  # Più neuroni
            activation='relu',
            max_iter=3000,  # Più iterazioni
            early_stopping=True,
            validation_fraction=0.15,
            alpha=0.001,  # Regolarizzazione
            learning_rate_init=0.001,
            random_state=42
        )
        model.fit(X_train_scaled, y_train_encoded)
        y_pred_encoded = model.predict(X_test_scaled)
        score = accuracy_score(y_test_encoded, y_pred_encoded)
        metric = "Accuracy"
        
        # NUOVO: Analisi dettagliata performance gear
        if target_name == 'gear':
            from sklearn.metrics import classification_report, confusion_matrix
            print(f"\n📈 Analisi dettagliata performance GEAR:")
            print("Classification Report:")
            gear_labels = encoder.inverse_transform(sorted(np.unique(y_train_encoded)))
            print(classification_report(y_test_encoded, y_pred_encoded, 
                                      target_names=[f"Gear_{g}" for g in gear_labels]))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test_encoded, y_pred_encoded)
            print("Pred\\Real", end="")
            for g in gear_labels:
                print(f"{g:>6}", end="")
            print()
            for i, g in enumerate(gear_labels):
                print(f"Gear {g:<4}", end="")
                for j in range(len(gear_labels)):
                    print(f"{cm[i,j]:>6}", end="")
                print()
    
    print(f"   {target_name}: {metric} = {score:.4f}")
    return target_name, model, scaler, encoder

def train_models(data_dir):
    """Training parallelo di tutti i modelli"""
    dataset = load_data(data_dir)
    
    models = {}
    scalers = {}
    encoders = {}
    
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
            target_name, model, scaler, encoder = future.result()
            models[target_name] = model
            if scaler:
                scalers[target_name] = scaler
            if encoder:
                encoders[target_name] = encoder
    
    print(f"\nTutti i modelli completati!")
    return models, scalers, encoders

def predict(sensor_data, models, scalers, encoders, steer_remap_factor=5):
    """
    Predice i controlli completi per un dato stato dei sensori
    
    Args:
        sensor_data (dict): Dati sensori TORCS
        models (dict): Modelli trainati
        scalers (dict): Scalers per neural networks
        encoders (dict): Encoders per classificazione
        steer_remap_factor (int): Fattore di remapping per steering
        
    Returns:
        dict: Controlli predetti (steer, throttle, brake, gear)
    """
    
    predictions = {}
    
    # ===============================
    # PREDIZIONE STEER (CON REMAPPING)
    # ===============================
    features_steer = FEATURE_SETS['steer']
    X_input_steer = np.array([[sensor_data[f] for f in features_steer]])
    X_input_steer = scalers['steer'].transform(X_input_steer)
    pred_steer_remapped = models['steer'].predict(X_input_steer)[0]
    
    # CRUCIAL: Inverse remapping to get actual steering angle
    pred_steer = pred_steer_remapped / steer_remap_factor
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
    # PREDIZIONE GEAR (CORRETTA)
    # ===============================
    features_gear = FEATURE_SETS['gear']
    X_input_gear = np.array([[sensor_data[f] for f in features_gear]])
    pred_gear_encoded = models['gear'].predict(X_input_gear)[0]
    pred_gear = encoders['gear'].inverse_transform([pred_gear_encoded])[0]
    predictions['gear'] = pred_gear
    
    return predictions

def save_model(models, scalers, encoders, filepath, steer_remap_factor=5):
    """Salva il modello trainato con informazioni di remapping"""
    model_data = {
        'models': models,
        'scalers': scalers,
        'encoders': encoders,
        'feature_sets': FEATURE_SETS,
        'optimal_models': OPTIMAL_MODELS,
        'steer_remap_factor': steer_remap_factor  # Save remapping factor
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Modello salvato in: {filepath}")

def load_model(filepath):
    """Carica un modello salvato con informazioni di remapping"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    models = model_data['models']
    scalers = model_data['scalers']
    encoders = model_data.get('encoders', {})
    steer_remap_factor = model_data.get('steer_remap_factor', 5)
    
    print(f"Modello caricato da: {filepath}")
    print(f"Steering remap factor: {steer_remap_factor}")
    return models, scalers, encoders

# ===============================
# TRAINING E TESTING
# ===============================

if __name__ == "__main__":

    import time
    start = time.time()

    # Training
    data_dir = './torcs_training_data'
    models, scalers, encoders = train_models(data_dir)
    
    # Salva il modello in una cartella con timestamp
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = f'models/model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'torcs_optimal_model.pkl')
    save_model(models, scalers, encoders, model_path)
    
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
    controls = predict(test_input, models, scalers, encoders)
    
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
    
    # NUOVO: Test predizioni gear su scenari multipli
    print("\n" + "="*50)
    print("TEST GEAR SU SCENARI MULTIPLI")
    print("="*50)
    
    test_scenarios = [
        {'speedX': 5, 'rpm': 2000, 'throttle': 0.8, 'brake': 0},    # Partenza
        {'speedX': 50, 'rpm': 4000, 'throttle': 0.5, 'brake': 0},   # Normale
        {'speedX': 100, 'rpm': 5000, 'throttle': 0.7, 'brake': 0},  # Velocità media
        {'speedX': 150, 'rpm': 6000, 'throttle': 0.9, 'brake': 0},  # Alta velocità
        {'speedX': 80, 'rpm': 7000, 'throttle': 0, 'brake': 0.8},   # Frenata
    ]
    
    print("Scenario                    | Speed | RPM  | Gear Pred")
    print("-" * 55)
    for i, scenario in enumerate(test_scenarios):
        # Completa con valori dummy per altri sensori
        full_scenario = {**test_input, **scenario}
        gear_pred = predict(full_scenario, models, scalers, encoders)['gear']
        print(f"Scenario {i+1:2d} ({['Partenza', 'Normale', 'Media', 'Alta', 'Frenata'][i]:<8}) | "
              f"{scenario['speedX']:>5.0f} | {scenario['rpm']:>4.0f} | {gear_pred:>8.0f}")
    
    print(f"\nModello ottimale pronto per l'uso!")
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)

    end = time.time()
    print(f"\nTempo totale di esecuzione: {end - start:.2f} secondi")
