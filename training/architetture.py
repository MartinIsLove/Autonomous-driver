import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

# ===============================
# CONFIGURAZIONE
# ===============================

# Feature sets identici al tuo training2.py
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

ARCHITECTURES_TO_TEST = [
    # ===== ARCHITETTURE BASE (3 LAYER) =====
    (64, 32, 16),
    (128, 64, 32),
    (256, 128, 64),
    (128, 128, 64),
    (64, 64, 32),
    (96, 48, 24),
    (160, 80, 40),
    (200, 100, 50),
    
    # ===== ARCHITETTURE MEDIE (4 LAYER) =====
    (128, 64, 32, 16),
    (256, 128, 64, 32),
    (128, 128, 64, 32),
    (200, 150, 100, 50),
    (256, 192, 128, 64),
    (320, 240, 160, 80),
    (128, 96, 64, 32),
    (180, 120, 80, 40),
    (224, 168, 112, 56),
    
    # ===== ARCHITETTURE PROFONDE (5 LAYER) =====
    (128, 64, 32, 16, 8),
    (256, 128, 64, 32, 16),
    (200, 150, 100, 50, 25),
    (256, 192, 128, 64, 32),
    (320, 240, 160, 80, 40),
    (128, 96, 64, 48, 32),
    (160, 120, 80, 60, 40),
    (240, 180, 120, 80, 40),
    
    # ===== ARCHITETTURE MOLTO PROFONDE (6+ LAYER) =====
    (256, 192, 128, 96, 64, 32),
    (320, 240, 160, 120, 80, 40),
    (128, 96, 72, 54, 40, 30),
    (200, 150, 112, 84, 63, 47),
    (256, 204, 163, 130, 104, 83),
    
    # ===== VARIANTI CON PLATEAU =====
    (128, 64, 32, 32, 32, 32, 8),
    (96, 48, 24, 24, 24, 24, 6),
    (80, 40, 20, 20, 20, 20, 5),
    
    # Decrescita più graduale
    (128, 96, 72, 54, 40, 30, 8),
    (160, 120, 90, 67, 50, 37, 10),
    (100, 80, 64, 51, 40, 32, 6),
    
    # Varianti con plateau diversi
    (64, 48, 32, 32, 24, 16, 4),
    (80, 60, 40, 40, 30, 20, 4),
    (96, 72, 48, 48, 36, 24, 4),
    
    # ===== ARCHITETTURE SPERIMENTALI =====
    # Con plateau iniziali
    (128, 128, 128, 64, 32, 16),
    (96, 96, 96, 48, 24, 12),
    (160, 160, 80, 80, 40, 20),
    
    # Con plateau finali
    (256, 128, 64, 32, 32, 32),
    (200, 100, 50, 25, 25, 25),
    (180, 90, 45, 22, 22, 22),
    
    # Forme a diamante
    (64, 128, 192, 128, 64, 32),
    (32, 64, 128, 64, 32, 16),
    (48, 96, 144, 96, 48, 24),
    
    # Architetture wide (molti neuroni nei primi layer)
    (512, 256, 128, 64, 32),
    (400, 300, 200, 100, 50),
    (384, 288, 192, 96, 48),
    (600, 450, 300, 150, 75),
    
    # Architetture narrow ma profonde
    (64, 32, 32, 32, 32, 32),
    (48, 32, 32, 32, 32, 32, 16),
    (40, 30, 30, 30, 30, 30, 20),
    (56, 42, 42, 42, 42, 28, 14),
    
    # Architetture con pattern specifici
    (120, 100, 80, 60, 40, 20),  # Decrescita lineare
    (128, 112, 96, 80, 64, 48),  # Decrescita costante
    (200, 160, 128, 102, 82, 66), # Decrescita fibonacci-like
    (144, 108, 81, 60, 45, 33),  # Decrescita 3/4
]

# Funzioni di attivazione da testare
ACTIVATIONS = ['tanh', 'relu', 'logistic']

# ===============================
# FUNZIONI
# ===============================

def load_data(data_dir):
    """Carica dati (identica al tuo training2.py)"""
    all_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    list_dataframes = []
    
    if not all_files:
        raise ValueError(f"Nessun file CSV trovato in: {data_dir}")
    
    print(f"Caricamento {len(all_files)} file CSV...")
    
    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            list_dataframes.append(df)
        except Exception as e:
            print(f"Errore {file}: {e}")
    
    dataset = pd.concat(list_dataframes, ignore_index=True)
    print(f"Dataset: {dataset.shape[0]} righe, {dataset.shape[1]} colonne")
    
    return dataset

def test_architecture(target_name, X_train, X_test, y_train, y_test, architecture, activation, max_iter=1000):
    """Testa una singola architettura"""
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modello
    model = MLPRegressor(
        hidden_layer_sizes=architecture,
        activation=activation,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        # solver='adam'
    )
    
    # Training
    start_time = time.time()
    try:
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Predizioni
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metriche
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Calcola numero parametri
        n_params = sum(layer.size for layer in model.coefs_) + sum(layer.size for layer in model.intercepts_)
        
        # Overfitting score (differenza train-test)
        overfitting = train_r2 - test_r2 if train_r2 > test_r2 else 0
        
        return {
            'architecture': architecture,
            'activation': activation,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'overfitting': overfitting,
            'training_time': training_time,
            'n_iterations': model.n_iter_,
            'n_parameters': n_params,
            'converged': model.n_iter_ < max_iter
        }
        
    except Exception as e:
        print(f"ERRORE con {architecture} + {activation}: {e}")
        return None

def test_architecture_wrapper(args):
    """Wrapper per permettere l'uso con ProcessPoolExecutor"""
    (target_name, X_train, X_test, y_train, y_test, architecture, activation, max_iter) = args
    return test_architecture(target_name, X_train, X_test, y_train, y_test, architecture, activation, max_iter)

def optimize_architectures(data_dir, n_workers=None):
    """Ottimizza architetture per tutti i target con processamento parallelo"""
    
    if n_workers is None:
        n_workers = multiprocessing.cpu_count() - 1  # Lascia un core libero
    
    print(f"🔄 Utilizzo {n_workers} worker paralleli")
    
    # Carica dati
    dataset = load_data(data_dir)
    
    # Risultati
    all_results = {}
    
    for target_name in ['steer']:
        print(f"\n{'='*60}")
        print(f"OTTIMIZZAZIONE ARCHITETTURA PER: {target_name.upper()}")
        print(f"{'='*60}")
        
        # Prepara dati
        X = dataset[FEATURE_SETS[target_name]]
        y = dataset[target_name]
        
        # Pulizia
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        print(f"Dati puliti: {len(X)} campioni")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Prepara argomenti per processamento parallelo
        tasks = []
        for arch, activation in product(ARCHITECTURES_TO_TEST, ACTIVATIONS):
            tasks.append((target_name, X_train, X_test, y_train, y_test, arch, activation, 1000))
        
        # Esecuzione parallela
        results = []
        total_tests = len(tasks)
        completed_tests = 0
        
        print(f"🚀 Avvio {total_tests} test in parallelo...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Sottometti tutti i task
            future_to_config = {
                executor.submit(test_architecture_wrapper, task): task 
                for task in tasks
            }
            
            # Raccogli risultati man mano che completano
            for future in as_completed(future_to_config):
                completed_tests += 1
                task = future_to_config[future]
                arch, activation = task[5], task[6]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"[{completed_tests:2d}/{total_tests:2d}] ✅ {arch} + {activation} → R²={result['test_r2']:.4f}")
                    else:
                        print(f"[{completed_tests:2d}/{total_tests:2d}] ❌ {arch} + {activation} → FAILED")
                except Exception as e:
                    print(f"[{completed_tests:2d}/{total_tests:2d}] ❌ {arch} + {activation} → ERROR: {e}")
        
        # Ordina per test R² score
        results.sort(key=lambda x: x['test_r2'], reverse=True)
        all_results[target_name] = results
        
        # Mostra top 5
        print(f"\n🏆 TOP 5 ARCHITETTURE per {target_name.upper()}:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Architecture':<25} {'Act':<8} {'Test R²':<8} {'Overfit':<8} {'Params':<8} {'Time':<6}")
        print("-" * 80)
        
        for i, result in enumerate(results[:5]):
            arch_str = str(result['architecture'])
            if len(arch_str) > 24:
                arch_str = arch_str[:21] + "..."
            
            print(f"{i+1:<4} {arch_str:<25} {result['activation']:<8} "
                  f"{result['test_r2']:<8.4f} {result['overfitting']:<8.4f} "
                  f"{result['n_parameters']:<8} {result['training_time']:<6.1f}s")
    
    return all_results

def save_results(results, output_dir='architecture_optimization_results'):
    """Salva risultati e genera grafici"""
    
    # Crea directory
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Salva JSON
    json_results = {}
    for target, target_results in results.items():
        json_results[target] = []
        for result in target_results:
            # Converti tuple in string per JSON
            json_result = result.copy()
            json_result['architecture'] = str(result['architecture'])
            json_results[target].append(json_result)
    
    with open(f"{full_output_dir}/results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Genera grafici
    plt.style.use('seaborn-v0_8')
    
    for target_name, target_results in results.items():
        if not target_results:
            continue
            
        # Top 10 results
        top_results = target_results[:10]
        
        # Grafico 1: R² Test Score
        plt.figure(figsize=(12, 6))
        arch_labels = [str(r['architecture'])[:15] + "..." if len(str(r['architecture'])) > 15 else str(r['architecture']) 
                      for r in top_results]
        test_scores = [r['test_r2'] for r in top_results]
        
        bars = plt.bar(range(len(top_results)), test_scores, color='skyblue')
        
        plt.title(f'Top 10 Architetture per {target_name.upper()} - Test R² Score')
        plt.xlabel('Architettura')
        plt.ylabel('R² Score')
        plt.xticks(range(len(top_results)), arch_labels, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{full_output_dir}/{target_name}_r2_scores.png", dpi=300)
        plt.close()
        
        # Grafico 2: Overfitting vs Performance
        plt.figure(figsize=(10, 6))
        overfit_scores = [r['overfitting'] for r in top_results]
        
        scatter = plt.scatter(test_scores, overfit_scores, c='blue', s=100, alpha=0.7)
        
        plt.title(f'{target_name.upper()} - Performance vs Overfitting')
        plt.xlabel('Test R² Score')
        plt.ylabel('Overfitting (Train R² - Test R²)')
        plt.grid(True, alpha=0.3)
        
        # Linea ideale (no overfitting)
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No Overfitting')
        
        # Annotazioni per top 3
        for i, result in enumerate(top_results[:3]):
            plt.annotate(f"#{i+1}", xy=(result['test_r2'], result['overfitting']), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{full_output_dir}/{target_name}_overfitting.png", dpi=300)
        plt.close()
    
    # Report finale
    report_path = f"{full_output_dir}/optimization_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("REPORT OTTIMIZZAZIONE ARCHITETTURE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Architetture testate: {len(ARCHITECTURES_TO_TEST)}\n")
        f.write(f"Funzioni attivazione: {ACTIVATIONS}\n\n")
        
        for target_name, target_results in results.items():
            f.write(f"\n{target_name.upper()}\n")
            f.write("-" * 30 + "\n")
            
            if target_results:
                best = target_results[0]
                f.write(f" MIGLIORE ARCHITETTURA:\n")
                f.write(f"   Architecture: {best['architecture']}\n")
                f.write(f"   Activation: {best['activation']}\n")
                f.write(f"   Test R²: {best['test_r2']:.6f}\n")
                f.write(f"   Overfitting: {best['overfitting']:.6f}\n")
                f.write(f"   Parametri: {best['n_parameters']}\n")
                f.write(f"   Tempo training: {best['training_time']:.2f}s\n\n")
                
                # Top 10
                f.write("🏅 TOP 10:\n")
                for i, result in enumerate(target_results[:10]):
                    f.write(f"   {i+1:2d}. {result['architecture']} + {result['activation']} "
                           f"(R²={result['test_r2']:.4f})\n")
    
    print(f"\n📁 Risultati salvati in: {full_output_dir}")
    print(f"📊 Grafici disponibili: *_r2_scores.png, *_overfitting.png")
    print(f"📋 Report completo: optimization_report.txt")
    
    return full_output_dir

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    
    # Necessario per Windows multiprocessing
    multiprocessing.freeze_support()
    
    print("🚀 OTTIMIZZAZIONE ARCHITETTURE NEURAL NETWORK (PARALLELA)")
    print("="*60)
    print(f"Architetture da testare: {len(ARCHITECTURES_TO_TEST)}")
    print(f"Funzioni attivazione: {len(ACTIVATIONS)}")
    print(f"Test totali per target: {len(ARCHITECTURES_TO_TEST) * len(ACTIVATIONS)}")
    print(f"CPU disponibili: {multiprocessing.cpu_count()}")
    print("="*60)
    
    start_time = time.time()
    
    # Ottimizzazione parallela
    data_dir = './torcs_training_data'
    results = optimize_architectures(data_dir)
    
    # Salva risultati
    output_dir = save_results(results)
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 OTTIMIZZAZIONE COMPLETATA!")
    print(f"⏱️  Tempo totale: {total_time:.1f} secondi")
    print(f"📁 Output salvato in: {output_dir}")
    
    # Suona quando finito
    try:
        import winsound
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
    except ImportError:
        pass
    
    # Riassunto finale
    print(f"\n{'='*60}")
    print("🏆 RIASSUNTO MIGLIORI ARCHITETTURE:")
    print(f"{'='*60}")
    
    for target_name, target_results in results.items():
        if target_results:
            best = target_results[0]
            print(f"{target_name.upper():>8}: {best['architecture']} + {best['activation']} "
                  f"(R²={best['test_r2']:.4f})")