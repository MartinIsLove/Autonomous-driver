import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from keras.metrics import RootMeanSquaredError, SparseCategoricalAccuracy
######################################
#preparazione dati
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Per salvare lo scaler
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
##################################

def create_driving_model(input_shape):
    """Crea un modello ibrido per controllo auto TORCS"""
    # creo il layer di input con 42 layer chiamato sensor_input
    inputs = Input(shape=input_shape, name='sensor_input')
    
    # Feature extraction condivisa
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    #############################
    # x = tf.expand_dims(x, axis=1)  # Add time dimension
    # x = tf.keras.layers.Reshape((1, 512))(x)
    # x = LSTM(256, return_sequences=False)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    ##################################
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Branch per regressione (sterzo, acceleratore, freno)
    reg_branch = Dense(64, activation='relu')(x)
    reg_branch = Dropout(0.1)(reg_branch)
    # reg_branch = Dense(64, activation='relu')(reg_branch)
    # reg_branch = Dense(32, activation='relu')(reg_branch)
    
    steer = Dense(1, activation='tanh', name='steer')(reg_branch)        # [-1, 1]
    throttle = Dense(1, activation='sigmoid', name='throttle')(reg_branch) # [0, 1]
    brake = Dense(1, activation='sigmoid', name='brake')(reg_branch)      # [0, 1]
    
    # Branch per classificazione (cambio marce)
    cls_branch = Dense(64, activation='relu')(x)
    # cls_branch = Dense(64, activation='relu')(cls_branch)
    # cls_branch = Dense(32, activation='relu')(cls_branch)
    cls_branch = Dropout(0.1)(cls_branch)
    gear = Dense(8, activation='softmax', name='gear')(cls_branch)  # 8 classi: [-1,0,1,2,3,4,5,6]
    
    return Model(inputs=inputs, outputs=[steer, throttle, brake, gear])

# Configurazione del modello
INPUT_SHAPE = (41,)  # 53 features dai sensori TORCS
model = create_driving_model(INPUT_SHAPE)

# Compilazione con loss e metriche specifiche

model.compile(
    # optimizer=Adam(learning_rate=0.0005),
    
    optimizer = RMSprop(learning_rate=0.0001),
    loss={
        'steer': MeanSquaredError(),
        'throttle': MeanSquaredError(),
        'brake': MeanSquaredError(),
        'gear': SparseCategoricalCrossentropy()
    },
    loss_weights={
        'steer': 5.0,
        'throttle': 1.0,
        'brake': 1.0,
        'gear': 1.0
    },
    metrics={
        'steer': RootMeanSquaredError(),
        'throttle': RootMeanSquaredError(),
        'brake': RootMeanSquaredError(),
        'gear': SparseCategoricalAccuracy()
    }
)

# Visualizza l'architettura
model.summary()

#################################################
#preparazione dati

def load_and_prepare_data(data_dir):
    """Carica e prepara i dati dai file CSV con struttura specificata"""
    # Carica tutti i file CSV nella directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Errore caricamento {file}: {e}")
    
    if not dfs:
        raise ValueError("Nessun dato CSV trovato nella directory specificata")
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    # Seleziona features e target
    # Features sensoriali (input del modello)
    sensor_features = [
        'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced',
        'fuel', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos', 'z',
        'lastLapTime'
    ]
    
    # Aggiungi sensori track (0-18)
    sensor_features += [f'track_{i}' for i in range(19)]
    
    # Aggiungi wheel spin velocity
    sensor_features += [f'wheelSpinVel_{i}' for i in range(4)]
    
    # Aggiungi focus sensors
    sensor_features += [f'focus_{i}' for i in range(5)]
    
    # Targets (output del modello)
    targets = {
        'steer': 'steer',
        'throttle': 'throttle',
        'brake': 'brake',
        'gear': 'gear'
    }
    
    # Features (X) - Dati sensoriali
    X = full_df[sensor_features].values
    
    # Targets (y) - Azioni del driver
    y_steer = full_df['steer'].values
    y_throttle = full_df['throttle'].values
    y_brake = full_df['brake'].values
    y_gear = full_df['gear'].values
    
    # Mappatura marce a classi: -1→0, 0→1, 1→2, ..., 6→7
    gear_mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
    y_gear_mapped = np.vectorize(gear_mapping.get)(y_gear)
    
    # Normalizzazione features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Salva lo scaler per uso futuro
    joblib.dump(scaler, 'torcs_scaler.save')
    
    return X_scaled, y_steer, y_throttle, y_brake, y_gear_mapped, sensor_features

def create_train_val_datasets(X, y_steer, y_throttle, y_brake, y_gear, test_size=0.2, random_state=42):
    """Crea dataset di training e validazione"""
    # Split dei dati
    X_train, X_val, y_steer_train, y_steer_val, y_throttle_train, y_throttle_val, \
    y_brake_train, y_brake_val, y_gear_train, y_gear_val = train_test_split(
        X, y_steer, y_throttle, y_brake, y_gear,
        test_size=test_size, 
        random_state=random_state
    )
    
    # Formatta per il modello
    
    train_data = {'sensor_input': X_train}
    train_labels = {
        'steer': y_steer_train,
        'throttle': y_throttle_train,
        'brake': y_brake_train,
        'gear': y_gear_train
    }
    
    val_data = {'sensor_input': X_val}
    val_labels = {
        'steer': y_steer_val,
        'throttle': y_throttle_val,
        'brake': y_brake_val,
        'gear': y_gear_val
    }
    
    return train_data, train_labels, val_data, val_labels

# Utilizzo
DATA_DIR = "torcs_training_data"

# Carica e prepara i dati
X, y_steer, y_throttle, y_brake, y_gear, feature_names = load_and_prepare_data(DATA_DIR)

# Stampa informazioni sui dati
print(f"Numero di features: {len(feature_names)}")
print(f"Esempio di features: {feature_names[:5]}...")
print(f"Dimensioni dataset: {X.shape[0]} campioni")

# Crea dataset per allenamento
train_data, train_labels, val_data, val_labels = create_train_val_datasets(
    X, y_steer, y_throttle, y_brake, y_gear
)

print("\nDivisione dataset:")
print(f"- Training: {len(train_data['sensor_input'])} campioni")
print(f"- Validazione: {len(val_data['sensor_input'])} campioni")


#############################################
#allenamento

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras
# Callbacks
callbacks = [
    # earlystopping termina l'allenamento se non ci sono miglioramenti e ripristina automaticamente i pesi migliori(della migliore epoca)
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_torcs_model.keras', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# Allenamento

history = model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=64,
    validation_data=(val_data, val_labels),
    callbacks=callbacks,
    verbose=1
)
# # Callbacks
# print("\n--- Training final model on all data ---")

# # Create a dataset with all available data
# all_data = {'sensor_input': X}
# all_labels = {
#     'steer': y_steer,
#     'throttle': y_throttle,
#     'brake': y_brake,
#     'gear': y_gear
# }

# # Configure callbacks for full data training
# final_callbacks = [
#     ModelCheckpoint('final_torcs_model.h5', save_best_only=True, monitor='loss'),
#     ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
# ]

# # Train on all data (no validation split)
# final_history = model.fit(
#     all_data,
#     all_labels,
#     epochs=50,  # You may want fewer epochs for the final training
#     batch_size=256,
#     callbacks=final_callbacks,
#     verbose=1
# )
# Salva il modello finale
# model.save('torcs_driver_model.h5')

keras.saving.save_model(model, 'torcs_driver_model.keras')
print("Modello salvato con successo!")