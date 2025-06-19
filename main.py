import time
import keyboard
from car import Car
from controls.ai_control import AIControl
from controls.keyboard_control import KeyboardControl
import data_logger # Aggiunto import

# --- Configuration ---
CONTROLLERS = {
    "ai": AIControl(model_path='models/model_10-06-2025_15-38-57/torcs_optimal_model.pkl'),
    "keyboard": KeyboardControl()
}
current_control_mode = "keyboard"
running = True

def switch_control_mode():
    """Switches to the next control mode."""
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
    """Main function to run the TORCS client."""
    global running
    car = Car()

    logging_enabled = False
    csv_filename = None
    log_choice = input("Vuoi attivare il logging dei dati per l'allenamento? (s/n): ").lower()
    if log_choice == 's':
        logging_enabled = True
        csv_filename = data_logger.setup_csv_logging()
        print(f"Logging telemetria attivo. I dati verranno salvati in {csv_filename}")

    keyboard.add_hotkey('+', switch_control_mode)
    keyboard.add_hotkey('esc', stop_running)
    print("Setup complete. Press '+' to switch control mode. Press 'esc' to exit.")

    if not car.connect():
        print("Failed to connect to TORCS.")
        return

    print(f"🏁 Starting control loop with '{current_control_mode.upper()}' mode.")
    try:
        while running:
            data_dict = car.receive_data()

            if data_dict:
                if data_dict.get('special') == 'shutdown':
                    print("\nShutdown signal from TORCS received.")
                    break
                if data_dict.get('special') == 'restart':
                    print("\n🔄 Restarting race...")
                    if hasattr(CONTROLLERS['ai'], 'prev_gear'):
                        CONTROLLERS['ai'].prev_gear = 3
                    continue

                controller = CONTROLLERS[current_control_mode]
                
                actions = controller.get_actions(game_state=car)

                # Log dei dati se abilitato
                if logging_enabled:
                    data_logger.log_telemetry(csv_filename, car.get_state_dict(), actions)
                
                car.set_controls(**actions)
                
                car.send_controls()
                
                print(f"\r📊 {car} | Mode: {current_control_mode.upper()}", end='', flush=True)

    except KeyboardInterrupt:
        print("\nUser interruption")
    except Exception as e:
        print(f"\nAn error occurred in the main loop: {e}")
    finally:
        print("\nDisconnecting from TORCS...")
        if car.connected:
            car.disconnect()
        keyboard.unhook_all()
        print("Session ended.")

if __name__ == "__main__":
    main()