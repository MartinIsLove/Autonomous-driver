# Ottimizzazioni per CPU all'inizio del file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'  # Usa tutti i core CPU

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(0)  # Usa tutti i core
tf.config.threading.set_inter_op_parallelism_threads(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Trovate {len(gpus)} GPU configurate")
else:
    print("Nessuna GPU trovata. TensorFlow verrà eseguito sulla CPU.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GRU, TimeDistributed, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError, SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_prepare_data



timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
TRAINING_DATA_DIR = "torcs_training_data"
DESTINATION_DIR = f"models/model_{timestamp}"
os.makedirs(DESTINATION_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10

X_train, Y_train = load_and_prepare_data(data_dir=TRAINING_DATA_DIR, destination_dir=DESTINATION_DIR)

driving_labels = {
    'steer': Y_train['steer'],
    'throttle': Y_train['throttle'],
    'brake': Y_train['brake'],
    'gear': Y_train['gear']
}

data_sequences = []
steer_targets = []
throttle_targets = []
brake_targets = []
gear_targets = []

for i in range(len(X_train) - SEQUENCE_LENGTH):
    data_sequences.append(X_train[i:i+SEQUENCE_LENGTH])
    steer_targets.append(driving_labels['steer'].iloc[i+SEQUENCE_LENGTH])
    throttle_targets.append(driving_labels['throttle'].iloc[i+SEQUENCE_LENGTH])
    brake_targets.append(driving_labels['brake'].iloc[i+SEQUENCE_LENGTH])
    gear_targets.append(driving_labels['gear'].iloc[i+SEQUENCE_LENGTH])

data_sequences = np.array(data_sequences)
steer_seq = np.array(steer_targets)
throttle_seq = np.array(throttle_targets)
brake_seq = np.array(brake_targets)
gear_seq = np.array(gear_targets)

print(f"Dati originali: {X_train.shape[0]} campioni")
print(f"Sequenze create: {data_sequences.shape[0]} sequenze")


train_seq, val_seq, train_steer, val_steer, train_throttle, val_throttle, train_brake, val_brake, train_gear, val_gear = train_test_split(
    data_sequences, 
    steer_seq, 
    throttle_seq, 
    brake_seq, 
    gear_seq, 
    test_size=0.2, 
    random_state=42
)


model = Sequential()
model.add(TimeDistributed(Dense(128, activation="relu"), input_shape=(SEQUENCE_LENGTH, X_train.shape[1])))
model.add(TimeDistributed(BatchNormalization()))
model.add(GRU(256, return_sequences=True, dropout=0.2))
model.add(GRU(128, dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(192, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="linear"))
model.add(Dense(4, activation="linear", name="outputs"))

# Visualizza l'architettura del modello
print("\n" + "="*80)
print("ARCHITETTURA DEL MODELLO")
print("="*80)
model.summary()
print("="*80 + "\n")


throttle_loss = lambda y_true, y_pred: tf.reduce_mean(tf.where((y_true > 0.8) | (y_true < 0.2), 5.0 * tf.abs(y_true - y_pred), tf.abs(y_true - y_pred)))

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)


training_callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(f"{DESTINATION_DIR}/best_model.keras", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
]

combined_targets = np.column_stack([train_steer, train_throttle, train_brake, train_gear])
val_combined_targets = np.column_stack([val_steer, val_throttle, val_brake, val_gear])

model.fit(
    train_seq,
    combined_targets,
    epochs=10,
    batch_size=128,
    validation_data=(val_seq, val_combined_targets),
    callbacks=training_callbacks
)


model.save(f"{DESTINATION_DIR}/final_model.keras")
print("Modello salvato!")