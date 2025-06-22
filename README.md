## Setup

### Prerequisiti

*   [Python 3.10+](https://www.python.org/)
*   [TORCS - The Open Racing Car Simulator](http://torcs.sourceforge.net/): È necessario avere TORCS installato e configurato.

### Installazione

1.  **Clona la repository:**
    ```bash
    git clone https://github.com/MartinIsLove/Autonomous-driver.git
    cd Autonomous-driver
    ```

2.  **Crea e attiva un ambiente virtuale (con Python 3.10):**
    #### Windows
    ```powershell
    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
    #### macOS/Linux
    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Installa i pacchetti Python richiesti:**
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo

### 1. Eseguire il Programma Principale

Per avviare la simulazione e l'agente di guida, esegui:
```bash
python main.py
```
All'avvio del programma:
1.  **Seleziona un modello:** Se sono disponibili modelli addestrati, puoi sceglierne uno. Premi Invio per selezionare il più recente.
2.  **Abilita il logging dei dati:** Premi `s` per attivare la registrazione della telemetria per il training, o `n` per disattivarla.

### Controlli in Gioco

*   **Tasto `Z`:** Cambia la modalità di controllo tra: `KEYBOARD`, `STEERING WHEEL` (se collegato), e `AI` (se un modello è stato caricato).
*   **Tasto `ESC`:** Termina il programma.

### 2. Allenare l'IA

Per addestrare un nuovo modello utilizzando i dati di telemetria registrati, esegui lo script di training:
```bash
python training/training.py
```
Questo creerà un nuovo modello nella cartella `models/`.

### 3. Visualizzare la Telemetria

Per vedere un grafico in tempo reale della telemetria dell'auto (velocità, sterzata, ecc.), avvia il monitor:
```bash
python monitor.py
```

### 4. Visualizzare i Dati Registrati

Per analizzare i dati di telemetria salvati in un file `.csv`, esegui il visualizzatore:
```bash
python visualizer.py
```










