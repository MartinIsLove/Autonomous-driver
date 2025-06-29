import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

class TorcsAutonomousDriver:
    """
    Sistema di guida autonoma ottimizzato per TORCS
    Basato sui migliori modelli identificati dal training
    """
    
    def __init__(self):
        # Configurazione feature per ogni controllo
        self.feature_sets = {
            'steer': [
                'angle', 'trackPos', 'speedX', 'speedY', 'speedZ',
                'track_0', 'track_1', 'track_2', 'track_3', 'track_4',
                'track_5', 'track_6', 'track_7', 'track_8', 'track_9',
                'track_10', 'track_11', 'track_12', 'track_13', 'track_14',
                'track_15', 'track_16', 'track_17', 'track_18'
            ],
            'throttle': [
                'speedX', 'angle', 'trackPos', 'rpm',
                'track_0', 'track_1', 'track_2'
            ],
            'brake': [
                'speedX', 'speedY', 'speedZ', 'angle', 'trackPos',
                'track_0', 'track_1', 'track_2'
            ],
            'gear': [
                'speedX', 'rpm'
            ]
        }
        
        # Modelli ottimali identificati
        self.optimal_models = {
            'steer': 'nn',      # Neural Network (R²: 0.8417)
            'throttle': 'nn',   # Neural Network (R²: 0.7748) 
            'brake': 'rf',      # Random Forest (R²: 0.4260)
            'gear': 'rf'        # Random Forest (Accuracy: 0.9780)
        }
        
        self.models = {}
        self.scalers = {}
        self.is_trained = False
    
    def load_data(self, data_dir):
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
    
    def train(self, data_dir):
        """Training completo con modelli ottimali"""
        
        # Carica dati
        dataset = self.load_data(data_dir)
        
        print("\n" + "="*50)
        print("TRAINING MODELLI OTTIMALI")
        print("="*50)
        
        for target_name in ['steer', 'throttle', 'brake', 'gear']:
            print(f"\nTraining {target_name.upper()}...")
            
            # Seleziona features specifiche
            X = dataset[self.feature_sets[target_name]]
            y = dataset[target_name]
            
            # Pulizia dati
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X, y = X[mask], y[mask]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Training del modello ottimale
            model_type = self.optimal_models[target_name]
            
            if target_name == 'gear':
                # Classificazione con Random Forest
                model = RandomForestClassifier(
                    n_estimators=150, 
                    max_depth=15, 
                    min_samples_split=5,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                metric_name = "Accuracy"
                
            elif model_type == 'nn':
                # Neural Network per steer e throttle
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32), 
                    activation='relu',
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                metric_name = "R²"
                
                self.scalers[target_name] = scaler
                
            else:  # Random Forest per brake
                model = RandomForestRegressor(
                    n_estimators=150, 
                    max_depth=15, 
                    min_samples_split=5,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                metric_name = "R²"
            
            self.models[target_name] = model
            print(f"   {model_type.upper()} - {metric_name}: {score:.4f}")
        
        self.is_trained = True
        print(f"\nModello di guida autonoma pronto!")
    
    def predict(self, sensor_data):
        """
        Predice i controlli completi per un dato stato dei sensori
        
        Args:
            sensor_data (dict): Dati sensori TORCS
            
        Returns:
            dict: Controlli predetti (steer, throttle, brake, gear)
        """
        
        if not self.is_trained:
            raise ValueError("Modello non ancora trainato! Usa .train() prima.")
        
        predictions = {}
        
        for target_name in ['steer', 'throttle', 'brake', 'gear']:
            # Estrai features necessarie
            features = self.feature_sets[target_name]
            X_input = np.array([[sensor_data[f] for f in features]])
            
            model_type = self.optimal_models[target_name]
            
            if model_type == 'nn':
                # Neural Network richiede scaling
                X_input = self.scalers[target_name].transform(X_input)
            
            # Predizione
            pred = self.models[target_name].predict(X_input)[0]
            
            # Post-processing per mantenere range validi
            if target_name == 'steer':
                pred = np.clip(pred, -1.0, 1.0)
            elif target_name in ['throttle', 'brake']:
                pred = np.clip(pred, 0.0, 1.0)
            elif target_name == 'gear':
                pred = int(max(-1, min(6, pred)))  # Gear range: -1 a 6
            
            predictions[target_name] = pred
        
        return predictions
    
    def save_model(self, filepath):
        """Salva il modello trainato"""
        if not self.is_trained:
            raise ValueError("Nessun modello da salvare")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_sets': self.feature_sets,
            'optimal_models': self.optimal_models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modello salvato in: {filepath}")
    
    def load_model(self, filepath):
        """Carica un modello salvato"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_sets = model_data['feature_sets']
        self.optimal_models = model_data['optimal_models']
        self.is_trained = True
        
        print(f"Modello caricato da: {filepath}")

# ===============================
# TRAINING E TESTING
# ===============================

if __name__ == "__main__":
    # Inizializza il sistema
    driver = TorcsAutonomousDriver()
    
    # Training
    data_dir = './torcs_training_data'
    driver.train(data_dir)
    
    # Salva il modello in una cartella con timestamp
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_dir = f'models/model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'torcs_optimal_model.pkl')
    driver.save_model(model_path)
    
    # Test con dati di esempio
    print("\n" + "="*50)
    print("TEST DEL MODELLO")
    print("="*50)
    
    # Carica dati per test
    test_data = driver.load_data(data_dir)
    sample_idx = 1000
    
    # Prepara input di test
    test_input = {}
    all_features = set()
    for features in driver.feature_sets.values():
        all_features.update(features)
    
    for feature in all_features:
        test_input[feature] = test_data[feature].iloc[sample_idx]
    
    # Predizione
    controls = driver.predict(test_input)
    
    print("🔍 Confronto Predizione vs Realtà:")
    print("-" * 40)
    for control in ['steer', 'throttle', 'brake', 'gear']:
        real_val = test_data[control].iloc[sample_idx]
        pred_val = controls[control]
        diff = abs(pred_val - real_val)
        
        print(f"{control:8} | Pred: {pred_val:6.3f} | Real: {real_val:6.3f} | Diff: {diff:6.3f}")
    
    print(f"\nModello ottimale pronto per l'uso!")