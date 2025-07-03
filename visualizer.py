import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurazione stile plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

data_dir = "./torcs_training_data"

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
    raise ValueError("Nessun dato CSV valido è stato caricato dalla directory specificata.")
    
full_dataframes = pd.concat(list_dataframes, ignore_index=True)
print(f"Dati concatenati. Shape totale: {full_dataframes.shape}")

# Normalizza i valori del freno durante il preprocessing
full_dataframes['brake_normalized'] = np.clip(full_dataframes['brake'] / 3.5, 0, 1)

# ANALISI E VISUALIZZAZIONE
print("\n" + "="*80)
print("ANALISI DATI TORCS")
print("="*80)

# 1. STATISTICHE GENERALI
print("\n1. STATISTICHE GENERALI:")
print("-" * 40)
print(f"Numero totale di campioni: {len(full_dataframes):,}")
print(f"Periodo di raccolta: {full_dataframes.shape[0]} frames")
print(f"Colonne disponibili: {full_dataframes.shape[1]}")

# 2. CONTROLLI (AZIONI DEL DRIVER)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CONTROLLI DEL VEICOLO', fontsize=16, fontweight='bold')

# Throttle
axes[0,0].hist(full_dataframes['throttle'], bins=50, alpha=0.7, color='green')
axes[0,0].set_title('Distribuzione Throttle')
axes[0,0].set_xlabel('Valore Throttle')
axes[0,0].axvline(full_dataframes['throttle'].mean(), color='red', linestyle='--', label=f'Media: {full_dataframes["throttle"].mean():.3f}')
axes[0,0].legend()

# Brake
axes[0,1].hist(full_dataframes['brake'], bins=50, alpha=0.7, color='red')
axes[0,1].set_title('Brake Distribution')
axes[0,1].set_xlabel('Brake Value')
axes[0,1].axvline(full_dataframes['brake'].mean(), color='blue', linestyle='--', label=f'Avarage: {full_dataframes["brake"].mean():.3f}')
axes[0,1].legend()

# Steer
axes[1,0].hist(full_dataframes['steer'], bins=50, alpha=0.7, color='blue')
axes[1,0].set_title('Distribuzione Steering')
axes[1,0].set_xlabel('Valore Steer')
axes[1,0].axvline(full_dataframes['steer'].mean(), color='red', linestyle='--', label=f'Media: {full_dataframes["steer"].mean():.3f}')
axes[1,0].legend()

# Gear
gear_counts = full_dataframes['gear'].value_counts().sort_index()
axes[1,1].bar(gear_counts.index, gear_counts.values, alpha=0.7, color='orange')
axes[1,1].set_title('Distribuzione Marce')
axes[1,1].set_xlabel('Marcia')
axes[1,1].set_ylabel('Frequenza')

plt.tight_layout()
plt.show()

# 3. VELOCITÀ E DINAMICA
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('DINAMICA DEL VEICOLO', fontsize=16, fontweight='bold')

# Velocità nel tempo (campione)
sample_size = min(5000, len(full_dataframes))
sample_data = full_dataframes.head(sample_size)

axes[0,0].plot(sample_data.index, sample_data['speedX'], alpha=0.7, color='purple')
axes[0,0].set_title('Velocità Longitudinale (Campione)')
axes[0,0].set_xlabel('Frame')
axes[0,0].set_ylabel('Velocità X')

# RPM
axes[0,1].hist(full_dataframes['rpm'], bins=50, alpha=0.7, color='red')
axes[0,1].set_title('Distribuzione RPM')
axes[0,1].set_xlabel('RPM')
axes[0,1].axvline(full_dataframes['rpm'].mean(), color='blue', linestyle='--', label=f'Media: {full_dataframes["rpm"].mean():.0f}')
axes[0,1].legend()

# Posizione sulla pista
axes[1,0].hist(full_dataframes['trackPos'], bins=50, alpha=0.7, color='green')
axes[1,0].set_title('Posizione sulla Pista')
axes[1,0].set_xlabel('Track Position')
axes[1,0].axvline(0, color='red', linestyle='--', label='Centro pista')
axes[1,0].legend()

# Fuel
axes[1,1].plot(sample_data.index, sample_data['fuel'], alpha=0.7, color='orange')
axes[1,1].set_title('Consumo Carburante (Campione)')
axes[1,1].set_xlabel('Frame')
axes[1,1].set_ylabel('Fuel')

plt.tight_layout()
plt.show()

# 4. SENSORI PISTA (Track sensors)
track_columns = [col for col in full_dataframes.columns if col.startswith('track_')]
if track_columns:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SENSORI DISTANZA PISTA', fontsize=16, fontweight='bold')
    
    # Heatmap sensori track
    track_data = full_dataframes[track_columns].head(1000)  # Primi 1000 campioni
    im = axes[0,0].imshow(track_data.T, aspect='auto', cmap='viridis')
    axes[0,0].set_title('Heatmap Sensori Track (1000 campioni)')
    axes[0,0].set_xlabel('Frame')
    axes[0,0].set_ylabel('Sensore Track')
    plt.colorbar(im, ax=axes[0,0])
    
    # Media sensori track
    track_means = full_dataframes[track_columns].mean()
    axes[0,1].bar(range(len(track_means)), track_means.values, alpha=0.7)
    axes[0,1].set_title('Distanza Media per Sensore')
    axes[0,1].set_xlabel('Indice Sensore')
    axes[0,1].set_ylabel('Distanza Media')
    
    # Boxplot alcuni sensori centrali
    central_sensors = track_columns[len(track_columns)//2-2:len(track_columns)//2+3]
    axes[1,0].boxplot([full_dataframes[col] for col in central_sensors])
    axes[1,0].set_title('Distribuzione Sensori Centrali')
    axes[1,0].set_xticklabels([col.split('_')[1] for col in central_sensors])
    
    # Anomalie (distanze molto piccole = vicino agli ostacoli)
    min_distances = full_dataframes[track_columns].min(axis=1)
    axes[1,1].hist(min_distances, bins=50, alpha=0.7, color='red')
    axes[1,1].set_title('Distanza Minima dagli Ostacoli')
    axes[1,1].set_xlabel('Distanza Minima')
    axes[1,1].axvline(5.0, color='orange', linestyle='--', label='Soglia Pericolo (5m)')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

# 5. CORRELAZIONI TRA VARIABILI
print("\n5. ANALISI CORRELAZIONI:")
print("-" * 40)

# Seleziona variabili principali per correlazione
main_vars = ['speedX', 'throttle', 'brake', 'steer', 'gear', 'rpm', 'trackPos', 'damage']
correlation_data = full_dataframes[main_vars]

plt.figure(figsize=(10, 8))
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice di Correlazione - Variabili Principali')
plt.tight_layout()
plt.show()

# 6. RILEVAMENTO ANOMALIE
print("\n6. RILEVAMENTO ANOMALIE:")
print("-" * 40)

anomalies = {}

# Velocità anomale
speed_q99 = full_dataframes['speedX'].quantile(0.99)
speed_anomalies = full_dataframes[full_dataframes['speedX'] > speed_q99]
anomalies['Velocità Extreme'] = len(speed_anomalies)

# Danni al veicolo
damage_anomalies = full_dataframes[full_dataframes['damage'] > 0]
anomalies['Danni Rilevati'] = len(damage_anomalies)

# Posizione fuori pista
off_track_anomalies = full_dataframes[abs(full_dataframes['trackPos']) > 1.0]
anomalies['Fuori Pista'] = len(off_track_anomalies)

# Throttle e brake contemporaneamente
throttle_brake_anomalies = full_dataframes[(full_dataframes['throttle'] > 0.5) & (full_dataframes['brake'] > 0.5)]
anomalies['Throttle+Brake Simultanei'] = len(throttle_brake_anomalies)

for anomaly_type, count in anomalies.items():
    percentage = (count / len(full_dataframes)) * 100
    print(f"{anomaly_type}: {count:,} campioni ({percentage:.2f}%)")

# 7. SUMMARY REPORT
print("\n" + "="*80)
print("REPORT FINALE")
print("="*80)

print(f"""
QUALITÀ DEI DATI:
- Campioni totali: {len(full_dataframes):,}
- Anomalie totali: {sum(anomalies.values()):,} ({sum(anomalies.values())/len(full_dataframes)*100:.2f}%)
- Utilizzo throttle medio: {full_dataframes['throttle'].mean():.2f}
- Utilizzo brake medio: {full_dataframes['brake'].mean():.2f}
- Steering medio: {full_dataframes['steer'].mean():.2f}

RACCOMANDAZIONI:
- {'✅ Dati bilanciati' if full_dataframes['throttle'].mean() > 0.3 else '⚠️  Poco throttle nei dati'}
- {'✅ Sterzate varie' if full_dataframes['steer'].std() > 0.1 else '⚠️  Poche sterzate nei dati'}
- {'✅ Pochi danni' if anomalies['Danni Rilevati'] < len(full_dataframes)*0.05 else '⚠️  Troppi danni rilevati'}
- {'✅ Guida in pista' if anomalies['Fuori Pista'] < len(full_dataframes)*0.1 else '⚠️  Spesso fuori pista'}
""")

# 8. MAPPA TRACCIATO E POSIZIONE VEICOLO
print("\n8. ANALISI TRACCIATO:")
print("-" * 40)

# Crea mappa del tracciato basata sui dati
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('.', fontsize=16, fontweight='bold')

# Calcola posizione assoluta del veicolo
# Usa distFromStart e trackPos per ricostruire il tracciato
sample_data = full_dataframes.iloc[::10]  # Campiona ogni 10 frame per performance

# Mappa 1: Tracciato ricostruito (vista dall'alto)
# Usa distFromStart come coordinata lungo il tracciato e trackPos come offset laterale
distance = sample_data['distFromStart']
lateral_pos = sample_data['trackPos']

# Stima coordinata X e Y (approssimazione)
# Assumiamo che il tracciato sia principalmente una curva
angle = distance / 1000  # Converti distanza in angolo approssimativo
x_track = np.cos(angle) * (1000 + lateral_pos * 10)  # Raggio base + offset laterale
y_track = np.sin(angle) * (1000 + lateral_pos * 10)

# Colora in base alla posizione: verde = in pista, rosso = fuori pista
colors = ['green' if abs(pos) <= 1.0 else 'red' for pos in lateral_pos]
axes[0,0].scatter(x_track, y_track, c=colors, alpha=0.6, s=1)
axes[0,0].set_title('Reconstructed Track (Green=On Track, Red=Off Track)')
axes[0,0].set_xlabel('X (meters)')
axes[0,0].set_ylabel('Y (meters)')
axes[0,0].axis('equal')

# Mappa 2: Posizione laterale nel tempo
time_sample = sample_data.index
axes[0,1].plot(time_sample, sample_data['trackPos'], alpha=0.7, color='blue')
axes[0,1].axhline(y=1.0, color='red', linestyle='--', label='Limite Pista Dx')
axes[0,1].axhline(y=-1.0, color='red', linestyle='--', label='Limite Pista Sx')
axes[0,1].axhline(y=0, color='green', linestyle='-', alpha=0.5, label='Centro Pista')
axes[0,1].fill_between(time_sample, -1.0, 1.0, alpha=0.2, color='green', label='Zona Sicura')
axes[0,1].set_title('Posizione Laterale nel Tempo')
axes[0,1].set_xlabel('Frame')
axes[0,1].set_ylabel('Track Position')
axes[0,1].legend()

# Mappa 3: Heatmap intensità posizione
# Crea griglia di posizioni
hist_2d, x_edges, y_edges = np.histogram2d(x_track, y_track, bins=50)
im = axes[1,0].imshow(hist_2d.T, origin='lower', cmap='hot', 
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[1,0].set_title('Heatmap Densità Posizioni')
axes[1,0].set_xlabel('X (metri)')
axes[1,0].set_ylabel('Y (metri)')
plt.colorbar(im, ax=axes[1,0], label='Frequenza')

# Mappa 4: Statistiche per settori del tracciato
# Divide il tracciato in settori basati su distFromStart
n_sectors = 10
full_dataframes['sector'] = pd.cut(full_dataframes['distFromStart'], 
                                   bins=n_sectors, labels=range(n_sectors))

# Calcola percentuale fuori pista per settore
sector_stats = []
for sector in range(n_sectors):
    sector_data = full_dataframes[full_dataframes['sector'] == sector]
    if len(sector_data) > 0:
        off_track_pct = (abs(sector_data['trackPos']) > 1.0).mean() * 100
        avg_speed = sector_data['speedX'].mean()
        sector_stats.append({'sector': sector, 'off_track_pct': off_track_pct, 'avg_speed': avg_speed})

if sector_stats:
    sector_df = pd.DataFrame(sector_stats)
    
    # Grafico a barre per settori problematici
    bars = axes[1,1].bar(sector_df['sector'], sector_df['off_track_pct'], 
                         color=['red' if x > 10 else 'green' for x in sector_df['off_track_pct']])
    axes[1,1].set_title('% Fuori Pista per Settore del Tracciato')
    axes[1,1].set_xlabel('Settore Tracciato')
    axes[1,1].set_ylabel('% Fuori Pista')
    axes[1,1].axhline(y=10, color='orange', linestyle='--', label='Soglia Critica (10%)')
    axes[1,1].legend()

plt.tight_layout()
plt.show()

# 9. ANALISI DETTAGLI FUORI PISTA
print("\n9. ANALISI DETTAGLIATA FUORI PISTA:")
print("-" * 40)

# Identifica episodi fuori pista
full_dataframes['off_track'] = abs(full_dataframes['trackPos']) > 1.0
full_dataframes['off_track_episode'] = (full_dataframes['off_track'] != 
                                       full_dataframes['off_track'].shift()).cumsum()

# Trova episodi consecutivi fuori pista
off_track_episodes = full_dataframes[full_dataframes['off_track']].groupby('off_track_episode').agg({
    'trackPos': ['min', 'max', 'mean'],
    'speedX': 'mean',
    'damage': 'max',
    'off_track': 'count'
}).round(3)

off_track_episodes.columns = ['TrackPos_Min', 'TrackPos_Max', 'TrackPos_Mean', 
                              'Speed_Avg', 'Max_Damage', 'Duration_Frames']

# Mostra episodi più lunghi
if len(off_track_episodes) > 0:
    print("EPISODI PIÙ LUNGHI FUORI PISTA:")
    long_episodes = off_track_episodes.nlargest(5, 'Duration_Frames')
    print(long_episodes)
    
    # Grafico episodi fuori pista
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(off_track_episodes['Duration_Frames'], bins=20, alpha=0.7, color='red')
    plt.title('Durata Episodi Fuori Pista')
    plt.xlabel('Durata (frames)')
    plt.ylabel('Frequenza')
    
    plt.subplot(2, 2, 2)
    plt.scatter(off_track_episodes['Speed_Avg'], off_track_episodes['Duration_Frames'], 
                alpha=0.6, color='blue')
    plt.title('Velocità vs Durata Fuori Pista')
    plt.xlabel('Velocità Media')
    plt.ylabel('Durata (frames)')
    
    plt.subplot(2, 2, 3)
    plt.hist(off_track_episodes['TrackPos_Mean'], bins=20, alpha=0.7, color='green')
    plt.title('Posizione Media Fuori Pista')
    plt.xlabel('Track Position')
    plt.ylabel('Frequenza')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Limite Dx')
    plt.axvline(x=-1.0, color='red', linestyle='--', label='Limite Sx')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    damage_episodes = off_track_episodes[off_track_episodes['Max_Damage'] > 0]
    if len(damage_episodes) > 0:
        plt.scatter(damage_episodes['Duration_Frames'], damage_episodes['Max_Damage'], 
                    alpha=0.6, color='red')
        plt.title('Durata vs Danni')
        plt.xlabel('Durata (frames)')
        plt.ylabel('Danni Massimi')
    else:
        plt.text(0.5, 0.5, 'Nessun danno rilevato\nnegli episodi fuori pista', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Durata vs Danni')
    
    plt.tight_layout()
    plt.show()

else:
    print("✅ Nessun episodio fuori pista rilevato!")

# Aggiorna il report finale con info mappa
print("\n" + "="*80)
print("REPORT FINALE AGGIORNATO")
print("="*80)

total_off_track = full_dataframes['off_track'].sum()
total_frames = len(full_dataframes)
off_track_percentage = (total_off_track / total_frames) * 100

print(f"""
QUALITÀ DEI DATI:
- Campioni totali: {len(full_dataframes):,}
- Frames fuori pista: {total_off_track:,} ({off_track_percentage:.2f}%)
- Episodi fuori pista: {len(off_track_episodes) if len(off_track_episodes) > 0 else 0}
- Utilizzo throttle medio: {full_dataframes['throttle'].mean():.2f}
- Utilizzo brake medio: {full_dataframes['brake'].mean():.2f}
- Steering medio: {full_dataframes['steer'].mean():.2f}

PERFORMANCE TRACCIATO:
- {'✅ Guida eccellente' if off_track_percentage < 5 else '⚠️ Guida da migliorare' if off_track_percentage < 15 else '❌ Guida problematica'}
- {'✅ Pochi episodi critici' if len(off_track_episodes) < 10 else '⚠️ Molti episodi fuori pista'}
- Media posizione: {full_dataframes['trackPos'].mean():.3f} (0 = centro perfetto)
- Deviazione standard: {full_dataframes['trackPos'].std():.3f}

RACCOMANDAZIONI:
- {'✅ Dati bilanciati' if full_dataframes['throttle'].mean() > 0.3 else '⚠️ Poco throttle nei dati'}
- {'✅ Sterzate varie' if full_dataframes['steer'].std() > 0.1 else '⚠️ Poche sterzate nei dati'}
- {'✅ Guida stabile' if off_track_percentage < 10 else '⚠️ Migliorare controllo veicolo'}
""")

# 10. VISUALIZZAZIONE SENSORI TRACK NEL TEMPO
print("\n10. SENSORI TRACK NEL TEMPO:")
print("-" * 40)

if track_columns:
    # Prendi un campione temporale più ampio per vedere l'evoluzione
    sample_frames = min(2000, len(full_dataframes))
    track_time_data = full_dataframes[track_columns].head(sample_frames)
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('ANALISI TEMPORALE SENSORI TRACK', fontsize=16, fontweight='bold')
    
    # 1. Heatmap temporale completa
    im1 = axes[0,0].imshow(track_time_data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=50)
    axes[0,0].set_title(f'Evoluzione Sensori Track ({sample_frames} frames)')
    axes[0,0].set_xlabel('Frame Temporale')
    axes[0,0].set_ylabel('Sensore (0=Sinistra, 18=Destra)')
    
    # Aggiungi etichette per i sensori
    sensor_labels = ['L9', 'L8', 'L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1', 'C', 
                     'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9']
    axes[0,0].set_yticks(range(0, 19, 2))
    axes[0,0].set_yticklabels([sensor_labels[i] for i in range(0, 19, 2)])
    plt.colorbar(im1, ax=axes[0,0], label='Distanza (m)')
    
    # 2. Sensori chiave nel tempo (linee)
    key_sensors = ['track_0', 'track_4', 'track_9', 'track_14', 'track_18']  # Sx, Sx-centro, Centro, Dx-centro, Dx
    sensor_names = ['Estrema Sx', 'Sx-Centro', 'Centro', 'Dx-Centro', 'Estrema Dx']
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, (sensor, name, color) in enumerate(zip(key_sensors, sensor_names, colors)):
        if sensor in track_time_data.columns:
            axes[0,1].plot(track_time_data.index, track_time_data[sensor], 
                          label=name, color=color, alpha=0.7, linewidth=1)
    
    axes[0,1].set_title('Sensori Chiave nel Tempo')
    axes[0,1].set_xlabel('Frame')
    axes[0,1].set_ylabel('Distanza (m)')
    axes[0,1].legend()
    axes[0,1].set_ylim(0, 50)  # Limita per vedere meglio i dettagli
    
    # 3. "Vista radar" - rappresentazione polare dei sensori in momenti specifici
    angles = np.linspace(-np.pi/2, np.pi/2, 19)  # Da -90° a +90°
    
    # Prendi alcuni frame rappresentativi
    sample_frames_idx = [100, 500, 1000, 1500]
    
    ax_polar = plt.subplot(3, 2, 3, projection='polar')
    for i, frame_idx in enumerate(sample_frames_idx):
        if frame_idx < len(track_time_data):
            distances = track_time_data.iloc[frame_idx].values
            distances = np.clip(distances, 0, 30)  # Limita per visualizzazione
            ax_polar.plot(angles, distances, 'o-', alpha=0.7, 
                         label=f'Frame {frame_idx}', linewidth=2)
    
    ax_polar.set_title('Vista Radar - Momenti Temporali')
    ax_polar.set_ylim(0, 30)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 4. Distanza minima nel tempo (pericolo)
    min_distances_time = track_time_data.min(axis=1)
    axes[1,1].plot(track_time_data.index, min_distances_time, 'r-', alpha=0.8)
    axes[1,1].axhline(y=5, color='orange', linestyle='--', label='Soglia Attenzione')
    axes[1,1].axhline(y=2, color='red', linestyle='--', label='Soglia Pericolo')
    axes[1,1].fill_between(track_time_data.index, 0, 2, alpha=0.3, color='red', label='Zona Pericolo')
    axes[1,1].fill_between(track_time_data.index, 2, 5, alpha=0.2, color='orange', label='Zona Attenzione')
    axes[1,1].set_title('Distanza Minima da Ostacoli nel Tempo')
    axes[1,1].set_xlabel('Frame')
    axes[1,1].set_ylabel('Distanza Minima (m)')
    axes[1,1].legend()
    
    # 5. Asimmetria sinistra-destra nel tempo
    left_sensors = track_time_data.iloc[:, :9].mean(axis=1)   # track_0 to track_8
    right_sensors = track_time_data.iloc[:, 10:].mean(axis=1)  # track_10 to track_18
    asymmetry = right_sensors - left_sensors
    
    axes[2,0].plot(track_time_data.index, asymmetry, 'b-', alpha=0.7)
    axes[2,0].axhline(y=0, color='green', linestyle='-', alpha=0.5, label='Simmetrico')
    axes[2,0].fill_between(track_time_data.index, 0, asymmetry, 
                          where=(asymmetry > 0), alpha=0.3, color='blue', 
                          label='Più spazio a destra')
    axes[2,0].fill_between(track_time_data.index, 0, asymmetry, 
                          where=(asymmetry < 0), alpha=0.3, color='red', 
                          label='Più spazio a sinistra')
    axes[2,0].set_title('Asimmetria Spazio Sinistra-Destra')
    axes[2,0].set_xlabel('Frame')
    axes[2,0].set_ylabel('Differenza Distanza (Dx-Sx)')
    axes[2,0].legend()
    
    # 6. Correlazione tra sensori e azioni di sterzo
    if 'steer' in full_dataframes.columns:
        steer_data = full_dataframes['steer'].head(sample_frames)
        
        # Calcola la "tendenza" dei sensori (dove c'è più spazio)
        sensor_bias = (right_sensors - left_sensors) / (right_sensors + left_sensors + 0.001)  # Normalizzato
        
        axes[2,1].scatter(sensor_bias, steer_data, alpha=0.5, s=1)
        axes[2,1].set_xlabel('Bias Sensori (>0 = più spazio a dx)')
        axes[2,1].set_ylabel('Sterzo (>0 = sterza a dx)')
        axes[2,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2,1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[2,1].set_title('Correlazione Sensori-Sterzo')
        
        # Calcola correlazione
        correlation = np.corrcoef(sensor_bias, steer_data)[0,1]
        axes[2,1].text(0.05, 0.95, f'Correlazione: {correlation:.3f}', 
                      transform=axes[2,1].transAxes, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # 11. ANALISI SITUAZIONI CRITICHE
    print("\n11. SITUAZIONI CRITICHE RILEVATE:")
    print("-" * 40)
    
    # Trova momenti di pericolo
    danger_threshold = 3.0
    danger_frames = track_time_data[min_distances_time < danger_threshold]
    
    if len(danger_frames) > 0:
        print(f"⚠️  {len(danger_frames)} frame con distanza < {danger_threshold}m")
        
        # Mostra i 5 momenti più pericolosi
        danger_moments = min_distances_time.nsmallest(5)
        print("\nMOMENTI PIÙ PERICOLOSI:")
        for frame_idx, distance in danger_moments.items():
            if frame_idx < len(full_dataframes):
                speed = full_dataframes.iloc[frame_idx]['speedX']
                steer = full_dataframes.iloc[frame_idx]['steer']
                print(f"Frame {frame_idx}: Distanza={distance:.2f}m, Velocità={speed:.2f}, Sterzo={steer:.3f}")
    else:
        print("✅ Nessuna situazione critica rilevata!")
    
    # Statistiche finali sensori
    print(f"\nSTATISTICHE SENSORI:")
    print(f"- Distanza media minima: {min_distances_time.mean():.2f}m")
    print(f"- Distanza minima assoluta: {min_distances_time.min():.2f}m")
    print(f"- Sensore sinistro medio: {left_sensors.mean():.2f}m")
    print(f"- Sensore destro medio: {right_sensors.mean():.2f}m")
    if 'steer' in full_dataframes.columns:
        print(f"- Correlazione sensori-sterzo: {correlation:.3f}")

# 12. ANALISI GIRI COMPLETATI
print("\n12. ANALISI GIRI COMPLETATI:")
print("-" * 40)

if 'lastLapTime' in full_dataframes.columns:
    # Rileva i cambi di giro quando lastLapTime cambia
    full_dataframes['lap_changed'] = full_dataframes['lastLapTime'] != full_dataframes['lastLapTime'].shift(1)
    full_dataframes['lap_number'] = full_dataframes['lap_changed'].cumsum()
    
    # Trova i punti di completamento giro (dove lastLapTime cambia)
    lap_completions = full_dataframes[full_dataframes['lap_changed'] & (full_dataframes.index > 0)]
    
    if len(lap_completions) > 0:
        print(f"🏁 {len(lap_completions)} giri completati rilevati")
        
        # Estrai i tempi dei giri
        lap_times = lap_completions['lastLapTime'].values
        lap_numbers = range(1, len(lap_times) + 1)
        
        # Calcola statistiche giri
        if len(lap_times) > 0:
            best_lap = min(lap_times)
            worst_lap = max(lap_times)
            avg_lap = np.mean(lap_times)
            
            print(f"- Miglior giro: {best_lap:.3f}s")
            print(f"- Peggior giro: {worst_lap:.3f}s") 
            print(f"- Tempo medio: {avg_lap:.3f}s")
            print(f"- Deviazione standard: {np.std(lap_times):.3f}s")
        
        # Visualizzazioni giri
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ANALISI GIRI COMPLETATI', fontsize=16, fontweight='bold')
        
        # 1. Tempi giri nel tempo
        axes[0,0].plot(lap_numbers, lap_times, 'o-', color='blue', linewidth=2, markersize=6)
        axes[0,0].axhline(y=best_lap, color='green', linestyle='--', label=f'Miglior: {best_lap:.3f}s')
        axes[0,0].axhline(y=avg_lap, color='orange', linestyle='--', label=f'Media: {avg_lap:.3f}s')
        axes[0,0].set_title('Tempi Giri nel Tempo')
        axes[0,0].set_xlabel('Numero Giro')
        axes[0,0].set_ylabel('Tempo Giro (s)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distribuzione tempi giri
        axes[0,1].hist(lap_times, bins=min(10, len(lap_times)), alpha=0.7, color='green', edgecolor='black')
        axes[0,1].axvline(best_lap, color='red', linestyle='--', label=f'Miglior: {best_lap:.3f}s')
        axes[0,1].axvline(avg_lap, color='blue', linestyle='--', label=f'Media: {avg_lap:.3f}s')
        axes[0,1].set_title('Distribuzione Tempi Giri')
        axes[0,1].set_xlabel('Tempo Giro (s)')
        axes[0,1].set_ylabel('Frequenza')
        axes[0,1].legend()
        
        # 3. Miglioramento nel tempo (trend)
        if len(lap_times) > 2:
            # Calcola trend lineare
            z = np.polyfit(lap_numbers, lap_times, 1)
            p = np.poly1d(z)
            axes[0,2].plot(lap_numbers, lap_times, 'o', color='blue', alpha=0.7, label='Tempi Giri')
            axes[0,2].plot(lap_numbers, p(lap_numbers), 'r--', linewidth=2, label=f'Trend: {z[0]:.4f}s/giro')
            axes[0,2].set_title('Trend Miglioramento')
            axes[0,2].set_xlabel('Numero Giro')
            axes[0,2].set_ylabel('Tempo Giro (s)')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            trend_text = "📈 Peggioramento" if z[0] > 0 else "📉 Miglioramento" if z[0] < 0 else "➡️ Stabile"
            axes[0,2].text(0.05, 0.95, trend_text, transform=axes[0,2].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 4. Analisi per giro - Velocità media
        lap_stats = []
        for lap_num in range(1, len(lap_completions) + 1):
            lap_data = full_dataframes[full_dataframes['lap_number'] == lap_num]
            if len(lap_data) > 0:
                lap_stats.append({
                    'lap': lap_num,
                    'lap_time': lap_times[lap_num-1] if lap_num-1 < len(lap_times) else 0,
                    'avg_speed': lap_data['speedX'].mean(),
                    'max_speed': lap_data['speedX'].max(),
                    'avg_throttle': lap_data['throttle'].mean(),
                    'avg_brake': lap_data['brake'].mean(),
                    'off_track_pct': (abs(lap_data['trackPos']) > 1.0).mean() * 100,
                    'damage': lap_data['damage'].max()
                })
        
        if lap_stats:
            lap_df = pd.DataFrame(lap_stats)
            
            # Velocità media per giro
            axes[1,0].bar(lap_df['lap'], lap_df['avg_speed'], alpha=0.7, color='purple')
            axes[1,0].set_title('Velocità Media per Giro')
            axes[1,0].set_xlabel('Numero Giro')
            axes[1,0].set_ylabel('Velocità Media')
            
            # Correlazione tempo giro vs velocità
            if len(lap_df) > 1:
                correlation_speed = np.corrcoef(lap_df['lap_time'], lap_df['avg_speed'])[0,1]
                axes[1,1].scatter(lap_df['avg_speed'], lap_df['lap_time'], 
                                 c=lap_df['lap'], cmap='viridis', s=100, alpha=0.7)
                axes[1,1].set_xlabel('Velocità Media')
                axes[1,1].set_ylabel('Tempo Giro (s)')
                axes[1,1].set_title(f'Velocità vs Tempo\n(Corr: {correlation_speed:.3f})')
                
                # Aggiungi colorbar per numero giro
                cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
                cbar.set_label('Numero Giro')
            
            # Percentuale fuori pista per giro
            colors_track = ['red' if x > 10 else 'orange' if x > 5 else 'green' for x in lap_df['off_track_pct']]
            axes[1,2].bar(lap_df['lap'], lap_df['off_track_pct'], alpha=0.7, color=colors_track)
            axes[1,2].set_title('% Fuori Pista per Giro')
            axes[1,2].set_xlabel('Numero Giro')
            axes[1,2].set_ylabel('% Fuori Pista')
            axes[1,2].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Soglia Critica')
            axes[1,2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 13. DETTAGLI GIRI SPECIFICI
        print("\n13. DETTAGLI GIRI SPECIFICI:")
        print("-" * 40)
        
        if len(lap_df) > 0:
            # Miglior e peggior giro
            best_lap_idx = lap_df['lap_time'].idxmin()
            worst_lap_idx = lap_df['lap_time'].idxmax()
            
            print(f"🥇 MIGLIOR GIRO #{lap_df.loc[best_lap_idx, 'lap']}:")
            print(f"   - Tempo: {lap_df.loc[best_lap_idx, 'lap_time']:.3f}s")
            print(f"   - Velocità media: {lap_df.loc[best_lap_idx, 'avg_speed']:.2f}")
            print(f"   - Throttle medio: {lap_df.loc[best_lap_idx, 'avg_throttle']:.3f}")
            print(f"   - Brake medio: {lap_df.loc[best_lap_idx, 'avg_brake']:.3f}")
            print(f"   - Fuori pista: {lap_df.loc[best_lap_idx, 'off_track_pct']:.1f}%")
            
            print(f"\n🥺 PEGGIOR GIRO #{lap_df.loc[worst_lap_idx, 'lap']}:")
            print(f"   - Tempo: {lap_df.loc[worst_lap_idx, 'lap_time']:.3f}s")
            print(f"   - Velocità media: {lap_df.loc[worst_lap_idx, 'avg_speed']:.2f}")
            print(f"   - Throttle medio: {lap_df.loc[worst_lap_idx, 'avg_throttle']:.3f}")
            print(f"   - Brake medio: {lap_df.loc[worst_lap_idx, 'avg_brake']:.3f}")
            print(f"   - Fuori pista: {lap_df.loc[worst_lap_idx, 'off_track_pct']:.1f}%")
            
            # Progressione delle performance
            if len(lap_df) >= 3:
                first_3_avg = lap_df.head(3)['lap_time'].mean()
                last_3_avg = lap_df.tail(3)['lap_time'].mean()
                improvement = first_3_avg - last_3_avg
                
                print(f"\n📊 PROGRESSIONE:")
                print(f"   - Media primi 3 giri: {first_3_avg:.3f}s")
                print(f"   - Media ultimi 3 giri: {last_3_avg:.3f}s")
                print(f"   - {'✅ Miglioramento' if improvement > 0 else '⚠️ Peggioramento'}: {abs(improvement):.3f}s")
        
        # 14. VISUALIZZAZIONE GIRI NEL TRACCIATO
        print("\n14. VISUALIZZAZIONE GIRI NEL TRACCIATO:")
        print("-" * 40)
        
        if len(lap_completions) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('ANALISI TRACCIATO PER GIRO', fontsize=16, fontweight='bold')
            
            # Mappa colorata per giro
            sample_data_laps = full_dataframes.iloc[::10]  # Campiona per performance
            distance = sample_data_laps['distFromStart']
            lateral_pos = sample_data_laps['trackPos']
            lap_colors = sample_data_laps['lap_number']
            
            # Ricostruisci tracciato
            angle = distance / 1000
            x_track = np.cos(angle) * (1000 + lateral_pos * 10)
            y_track = np.sin(angle) * (1000 + lateral_pos * 10)
            
            scatter = axes[0,0].scatter(x_track, y_track, c=lap_colors, cmap='tab10', 
                                       alpha=0.6, s=2)
            axes[0,0].set_title('Tracciato Colorato per Giro')
            axes[0,0].set_xlabel('X (metri)')
            axes[0,0].set_ylabel('Y (metri)')
            axes[0,0].axis('equal')
            plt.colorbar(scatter, ax=axes[0,0], label='Numero Giro')
            
            # Posizione laterale per giro (solo primi 5 giri per chiarezza)
            max_laps_to_show = min(5, int(full_dataframes['lap_number'].max()))
            colors_laps = plt.cm.tab10(np.linspace(0, 1, max_laps_to_show))
            
            for lap_num in range(1, max_laps_to_show + 1):
                lap_data = full_dataframes[full_dataframes['lap_number'] == lap_num]
                if len(lap_data) > 0:
                    # Normalizza gli indici per ogni giro per confronto
                    lap_progress = np.linspace(0, 100, len(lap_data))
                    axes[0,1].plot(lap_progress, lap_data['trackPos'], 
                                  label=f'Giro {lap_num}', alpha=0.7, 
                                  color=colors_laps[lap_num-1])
            
            axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            axes[0,1].axhline(y=-1.0, color='red', linestyle='--', alpha=0.5)
            axes[0,1].axhline(y=0, color='green', linestyle='-', alpha=0.3)
            axes[0,1].set_title('Posizione Laterale per Giro (primi 5)')
            axes[0,1].set_xlabel('Progresso Giro (%)')
            axes[0,1].set_ylabel('Track Position')
            axes[0,1].legend()
            
            # Velocità per giro
            for lap_num in range(1, max_laps_to_show + 1):
                lap_data = full_dataframes[full_dataframes['lap_number'] == lap_num]
                if len(lap_data) > 0:
                    lap_progress = np.linspace(0, 100, len(lap_data))
                    axes[1,0].plot(lap_progress, lap_data['speedX'], 
                                  label=f'Giro {lap_num}', alpha=0.7,
                                  color=colors_laps[lap_num-1])
            
            axes[1,0].set_title('Velocità per Giro (primi 5)')
            axes[1,0].set_xlabel('Progresso Giro (%)')
            axes[1,0].set_ylabel('Velocità X')
            axes[1,0].legend()
            
            # Confronto settori tra giri
            n_sectors = 5
            sector_analysis = []
            
            for lap_num in range(1, int(full_dataframes['lap_number'].max()) + 1):
                lap_data = full_dataframes[full_dataframes['lap_number'] == lap_num]
                if len(lap_data) > 10:  # Solo giri con dati sufficienti
                    # Dividi il giro in settori
                    sector_size = len(lap_data) // n_sectors
                    for sector in range(n_sectors):
                        start_idx = sector * sector_size
                        end_idx = (sector + 1) * sector_size if sector < n_sectors - 1 else len(lap_data)
                        sector_data = lap_data.iloc[start_idx:end_idx]
                        
                        sector_analysis.append({
                            'lap': lap_num,
                            'sector': sector + 1,
                            'avg_speed': sector_data['speedX'].mean(),
                            'time': len(sector_data) / 50.0  # Assume 50 FPS
                        })
            
            if sector_analysis:
                sector_df = pd.DataFrame(sector_analysis)
                sector_pivot = sector_df.pivot(index='lap', columns='sector', values='avg_speed')
                
                im = axes[1,1].imshow(sector_pivot.values, aspect='auto', cmap='RdYlGn')
                axes[1,1].set_title('Velocità Media per Settore')
                axes[1,1].set_xlabel('Settore')
                axes[1,1].set_ylabel('Numero Giro')
                axes[1,1].set_xticks(range(n_sectors))
                axes[1,1].set_xticklabels([f'S{i+1}' for i in range(n_sectors)])
                plt.colorbar(im, ax=axes[1,1], label='Velocità Media')
            
            plt.tight_layout()
            plt.show()
    
    else:
        print("❌ Nessun giro completato rilevato nei dati")
        print("   Verifica che 'lastLapTime' cambi durante la sessione")

else:
    print("❌ Colonna 'lastLapTime' non trovata nei dati")
    print("   Assicurati che i dati TORCS includano le informazioni sui giri")