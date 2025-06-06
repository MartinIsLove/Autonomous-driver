import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, Attention
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from keras.metrics import RootMeanSquaredError, SparseCategoricalAccuracy
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from keras.layers import LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras

def create_driving_model(input_shape, n_steps=10):
    inputs = Input(shape=(n_steps, input_shape[0]), name='sensor_input')
    
    # Preprocessing sequenziale
    x = tf.keras.layers.TimeDistributed(Dense(128, activation='relu'))(inputs)
    x = tf.keras.layers.TimeDistributed(BatchNormalization())(x)
    
    # Enhanced recurrent layers with stateful processing
    # First GRU layer with return sequences to maintain temporal information
    x = GRU(256, return_sequences=True, dropout=0.2, 
            recurrent_dropout=0.1, 
            recurrent_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    
    # Add attention mechanism to focus on important time steps
    attention = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Concatenate()([x, attention])
    
    # Second GRU layer for final sequence processing
    x = GRU(128, dropout=0.2, recurrent_dropout=0.1)(x)
    x = BatchNormalization()(x)
    
    # Skip Connection per preservare informazioni temporali
    shortcut = x
    x = Dense(192, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Concatenate()([x, shortcut])  # Skip connection
    
    # Branch per regressione con constraint fisici
    reg_branch = Dense(128, activation='relu')(x)
    reg_branch = Dense(64, activation='linear')(reg_branch)
    
    # Output con attivazioni specializzate
    steer = Dense(1, activation='tanh', name='steer')(reg_branch)
    
    # Throttle e Brake con interazione fisica
    throttle_input = Concatenate()([reg_branch, x])  # Maggior contesto
    throttle = Dense(1, activation='sigmoid', name='throttle')(throttle_input)
    
    brake_input = Concatenate()([reg_branch, x])
    brake = Dense(1, activation='sigmoid', name='brake')(brake_input)
    
    # Branch per cambio marce con regolarizzazione
    cls_branch = Dense(128, activation='relu')(x)
    cls_branch = Dropout(0.3)(cls_branch)
    cls_branch = Dense(64, activation='relu')(cls_branch)
    gear = Dense(8, activation='softmax', name='gear')(cls_branch)
    
    return Model(inputs=inputs, outputs=[steer, throttle, brake, gear])

def create_sequences(data, targets, n_steps):
    """Transform data into sequences for time-series processing"""
    X_seq = []
    y_steer_seq = []
    y_throttle_seq = []
    y_brake_seq = []
    y_gear_seq = []
    
    for i in range(len(data) - n_steps):
        # Create sequence of n_steps
        X_seq.append(data[i:i+n_steps])
        
        # Target is the next action after the sequence
        y_steer_seq.append(targets['steer'][i+n_steps])
        y_throttle_seq.append(targets['throttle'][i+n_steps])
        y_brake_seq.append(targets['brake'][i+n_steps])
        y_gear_seq.append(targets['gear'][i+n_steps])
    
    return np.array(X_seq), np.array(y_steer_seq), np.array(y_throttle_seq), np.array(y_brake_seq), np.array(y_gear_seq)

def adaptive_throttle_loss(y_true, y_pred):
    # Penalizza maggiormente gli errori in situazioni critiche
    error = tf.abs(y_true - y_pred)
    
    # Aumenta il peso quando:
    # - Il throttle reale è alto (accelerazione importante)
    # - Il throttle reale è basso (decelerazione/frenata)
    weights = tf.where(
        (y_true > 0.8) | (y_true < 0.2),
        5.0,  # Situazioni critiche
        1.0   # Situazioni normali
    )
    return tf.reduce_mean(weights * error)

def load_and_prepare_data(data_dir):
    """Carica e prepara i dati dai file CSV con struttura specificata"""
    # Carica tutti i file CSV nella directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
            # Skip empty DataFrames
            if df.empty:
                print(f"Skipping empty file: {file}")
                continue
                
            # Drop columns that are all NaN
            df = df.dropna(axis=1, how='all')
            
            # Only add non-empty DataFrames to the list
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Errore caricamento {file}: {e}")
    
    if not dfs:
        raise ValueError("Nessun dato CSV trovato nella directory specificata")
    
    # Ensure all DataFrames have the same columns
    common_columns = set.intersection(*[set(df.columns) for df in dfs])
    dfs = [df[list(common_columns)] for df in dfs]
    
    # Now concatenate with ignore_index
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
    
    # Ensure all required columns exist
    for feature in sensor_features:
        if feature not in full_df.columns:
            print(f"Warning: {feature} not found in data, initializing with zeros")
            full_df[feature] = 0.0
    
    # Ensure target columns exist
    target_cols = ['steer', 'throttle', 'brake', 'gear']
    for col in target_cols:
        if col not in full_df.columns:
            raise ValueError(f"Target column {col} not found in data")
    
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

class CarStatePredictor:
    def __init__(self, model, n_steps, scaler=None):
        self.model = model
        self.n_steps = n_steps
        self.state_buffer = None
        self.scaler = scaler
        
    def initialize_state(self, initial_sequence):
        """Initialize the state buffer with a sequence of observations"""
        if initial_sequence.shape[0] != self.n_steps:
            raise ValueError(f"Initial sequence must have {self.n_steps} time steps")
        
        # Apply scaling if scaler is provided
        if self.scaler and len(initial_sequence.shape) == 2:
            self.state_buffer = np.array([self.scaler.transform(initial_sequence)])
        else:
            self.state_buffer = initial_sequence.copy()
        
    def predict_next(self, new_observation):
        """Make prediction and update state"""
        if self.state_buffer is None:
            raise ValueError("State buffer not initialized. Call initialize_state first.")
        
        # Scale new observation if scaler is provided
        if self.scaler:
            new_observation = self.scaler.transform(np.array([new_observation]))[0]
            
        # Update state buffer (remove oldest, add newest)
        self.state_buffer = np.roll(self.state_buffer, -1, axis=1)
        self.state_buffer[:, -1] = new_observation
        
        # Get prediction
        predictions = self.model.predict(self.state_buffer)
        
        return {
            'steer': predictions[0][0][0],
            'throttle': predictions[1][0][0],
            'brake': predictions[2][0][0],
            'gear': np.argmax(predictions[3][0])
        }

# Main execution code
if __name__ == "__main__":
    # Configurazione del modello
    N_STEPS = 10
    DATA_DIR = "torcs_training_data"
    
    # Carica e prepara i dati
    X, y_steer, y_throttle, y_brake, y_gear, feature_names = load_and_prepare_data(DATA_DIR)
    
    # Create target dictionary for sequence creation
    targets = {
        'steer': y_steer,
        'throttle': y_throttle,
        'brake': y_brake,
        'gear': y_gear
    }
    
    # Create sequences for time-series modeling
    X_seq, y_steer_seq, y_throttle_seq, y_brake_seq, y_gear_seq = create_sequences(
        X, targets, N_STEPS
    )
    
    # Split in train and validation
    X_train, X_val, y_steer_train, y_steer_val, y_throttle_train, y_throttle_val, \
    y_brake_train, y_brake_val, y_gear_train, y_gear_val = train_test_split(
        X_seq, y_steer_seq, y_throttle_seq, y_brake_seq, y_gear_seq,
        test_size=0.2, 
        random_state=42
    )
    
    # Format data for the model
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
    
    # Stampa informazioni sui dati
    print(f"Numero di features: {len(feature_names)}")
    print(f"Esempio di features: {feature_names[:5]}...")
    print(f"Dimensioni dataset originale: {X.shape[0]} campioni")
    print(f"Dimensioni dataset sequenze: {X_seq.shape[0]} sequenze")
    
    print("\nDivisione dataset:")
    print(f"- Training: {len(train_data['sensor_input'])} sequenze")
    print(f"- Validazione: {len(val_data['sensor_input'])} sequenze")
    
    # Create and compile model
    INPUT_SHAPE = (X.shape[1],)  # Numero di features
    model = create_driving_model(INPUT_SHAPE, n_steps=N_STEPS)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss={
            'steer': MeanSquaredError(),
            'throttle': adaptive_throttle_loss,
            'brake': MeanSquaredError(),
            'gear': SparseCategoricalCrossentropy()
        },
        loss_weights={
            'steer': 4.0,
            'throttle': 3.0,
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
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_torcs_model.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
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
    
    # Salva il modello finale
    keras.saving.save_model(model, 'torcs_driver_model.keras')
    print("Modello salvato con successo!")
    
    # Esempio di creazione di un predittore stateful
    print("\nCreazione predittore stateful per inferenza...")
    scaler = joblib.load('torcs_scaler.save')
    predictor = CarStatePredictor(model, N_STEPS, scaler)
    print("Predittore stateful creato!")