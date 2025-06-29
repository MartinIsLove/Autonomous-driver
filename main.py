import os
import time
import atexit
import keyboard
import subprocess
from datetime import datetime

from car import Car

from controls.ai_control import AIControl
from controls.keyboard_control import KeyboardControl
# from controls.steering_wheel_control import SteeringWheelControl
# from controls.gamepad_controll import DualShockControl

import data_logger 


CONTROLLERS = {}
current_control_mode = "keyboard"
running = True
torcs_process = None

def start_torcs():
    """Avvia l'eseguibile di TORCS"""
    global torcs_process
    torcs_exe_path = os.path.join('torcs', 'torcs-exe', 'wtorcs.exe')
    torcs_dir = os.path.dirname(torcs_exe_path)

    if not os.path.exists(torcs_exe_path):
        print(f"Eseguibile di TORCS non trovato in: {torcs_exe_path}")
        return False

    print("Avvio di TORCS in corso..")
    try:
        torcs_process = subprocess.Popen(
            [os.path.abspath(torcs_exe_path)], 
            cwd=os.path.abspath(torcs_dir),
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        print("TORCS avviato con successo")
        time.sleep(5)  # Attendi che TORCS si inizializzi
        return True
    except Exception as e:
        print(f"Impossibile avviare TORCS: {e}")
        return False

def stop_torcs():
    """Arresta il processo di TORCS se è in esecuzione"""
    global torcs_process
    if torcs_process:
        print("Arresto di TORCS in corso..")
        torcs_process.terminate()
        torcs_process.wait()
        print("TORCS arrestato")

def select_model():
    """Permette all'utente di selezionare un modello allenato"""
    models_dir = 'models'
    try:
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('model_')]
    except FileNotFoundError:
        print(f"La cartella '{models_dir}' non è stata trovata")
        return None

    if not available_models:
        print("Nessun modello trovato. Eseguire prima il training")
        return None

    # Ordina i modelli per data, dal più recente al più vecchio
    try:
        available_models.sort(key=lambda x: datetime.strptime(x.replace('model_', ''), '%d-%m-%Y_%H-%M-%S'), reverse=True)
    except ValueError as e:
        print(f"Errore nel parsing dei nomi delle cartelle dei modelli: {e}")
        print("Assicurati che i nomi delle cartelle seguano il formato 'model_gg-mm-aaaa_hh-mm-ss'")
        return None

    print("\n--- Seleziona un modello ---")
    for i, model_name in enumerate(available_models):
        print(f"[{i+1}] {model_name}" + (" (ultimo allenato)" if i == 0 else ""))

    while True:
        try:
            choice = input(f"Inserisci il numero del modello (1-{len(available_models)}) o premi Invio per usare l'ultimo: ")
            if not choice:
                selected_model_name = available_models[0]
                break
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model_name = available_models[choice_idx]
                break
            else:
                print("Scelta non valida")
        except ValueError:
            print("Inserisci un numero")

    model_path = os.path.join(models_dir, selected_model_name, 'torcs_optimal_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Errore: file del modello non trovato in '{model_path}'")
        return None
        
    print(f"Modello selezionato: {model_path}")
    return model_path

def switch_control_mode():
    """Switches to the next control mode"""
    global current_control_mode
    modes = list(CONTROLLERS.keys())
    current_index = modes.index(current_control_mode)
    next_index = (current_index + 1) % len(modes)
    current_control_mode = modes[next_index]
    print(f"\n--- Control mode switched to: {current_control_mode.upper()} ---")

def stop_running():
    global running
    running = False
    print("\n--- Stop signal received ---")

def main():
    """Main function to run the TORCS client"""
    global running, CONTROLLERS, current_control_mode

    atexit.register(stop_torcs)

    if not start_torcs():
        return

    model_path = select_model()

    CONTROLLERS["keyboard"] = KeyboardControl()

    # try:
    #     CONTROLLERS["gamepad"] = DualShockControl()
    # except IOError as e:
    #     print(f"Attenzione: {e}. Il controller non sarà disponibile")

    # try:
    #     CONTROLLERS["steering_wheel"] = SteeringWheelControl()
    # except IOError as e:
    #     print(f"Attenzione: {e}. Il controllo da volante non sarà disponibile")
   
    if model_path:
        CONTROLLERS["ai"] = AIControl(model_path=model_path)
    
    current_control_mode = "keyboard"
    
    car = Car()

    logging_enabled = False
    csv_filename = None
    log_choice = input("Vuoi attivare il logging dei dati per l'allenamento? (s/N): ").lower()

    if log_choice == 's':

        logging_enabled = True
        csv_filename = data_logger.setup_csv_logging()
        
        print(f"Logging telemetria attivo:  {csv_filename}")

    keyboard.add_hotkey('z', switch_control_mode)
    keyboard.add_hotkey('esc', stop_running)

    print("Premi Z per cambiare input. Esc to exit")

    if not car.connect():
        print("Failed to connect to TORCS")
        return

    print(f"Input: '{current_control_mode.upper()}'")
    try:
        while running:
            data_dict = car.receive_data()

            if data_dict:
                if data_dict.get('special') == 'shutdown':
                    print("\nSTOP")
                    break
                if data_dict.get('special') == 'restart':
                    print("\nRESTART")
                    if hasattr(CONTROLLERS['ai'], 'prev_gear'):
                        CONTROLLERS['ai'].prev_gear = 3
                    continue

                controller = CONTROLLERS[current_control_mode]
                actions = controller.get_actions(game_state=car)

                if logging_enabled:
                    data_logger.log_telemetry(csv_filename, car.get_state_dict(), actions)
                
                car.set_controls(**actions)
                
                car.send_controls()
                
                print(f"\r📊 {car} | Mode: {current_control_mode.upper()}", end='', flush=True)

    except KeyboardInterrupt:
        
        print("\nEsc premuto")
    except Exception as e:
        print(f"{e}")
    finally:
        print("\nDisconnesisone TORCS..")
        if car.connected:
            car.disconnect()
        keyboard.unhook_all()

if __name__ == "__main__":
    main()