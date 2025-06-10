import math
import pygame
import numpy as np

def get_joystick_controls(controls, joystick, prev_gear_up, prev_gear_down):
    pygame.event.pump()
    raw_steer = joystick.get_axis(0) * -1
    steer = max(-1.0, min(1.0, raw_steer * 1.5))
    
    throttle = (1 - joystick.get_axis(1)) / 2
    
    raw_brake = (1 - joystick.get_axis(2)) / 2
    brake = 1 - np.log(1 + 180 * (1 - raw_brake)) / np.log(1 + 180)

    gear_up = joystick.get_button(4)
    gear_down = joystick.get_button(5)
    
    controls['steer'] = steer
    controls['throttle'] = throttle
    controls['brake'] = brake
    
    if gear_up and not prev_gear_up and controls['gear'] < 6:
        controls['gear'] += 1
    if gear_down and not prev_gear_down and controls['gear'] > -1:
        controls['gear'] -= 1
    return controls, gear_up, gear_down

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Simula la corsa reale del pedale freno (premuto = 1, non premuto = 0)
    raw_brake = np.linspace(0, 1, 100)
    # Curva logaritmica invertita: parte da 0,0 e arriva a 1,1 con la gobba in basso
    # CORREZIONE: Usa np.log() invece di math.log() per operazioni su array
    # brake_modulated = 1 - np.log(1 + 100 * (1 - raw_brake)) / np.log(1 + 100)
    
    brake_modulated_01 = 1 - np.log(1 + 50 * (1 - raw_brake)) / np.log(1 + 50)
    brake_modulated_02 = 1 - np.log(1 + 100 * (1 - raw_brake)) / np.log(1 + 100)
    brake_modulated_03 = 1 - np.log(1 + 180 * (1 - raw_brake)) / np.log(1 + 180)
    
    
    plt.figure(figsize=(7, 5))
    plt.plot(raw_brake, brake_modulated_01, label="Curva freno_01 (logaritmica invertita)")
    plt.plot(raw_brake, brake_modulated_02, label="Curva freno_02 (logaritmica invertita)")
    plt.plot(raw_brake, brake_modulated_03, label="Curva freno_03 (logaritmica invertita)")
    plt.plot(raw_brake, raw_brake, '--', label="Lineare (nessuna modulazione)")
    plt.title("Risposta del freno: pedale reale vs valore inviato al gioco")
    plt.xlabel("Pedale freno reale (non premuto → premuto)")
    plt.ylabel("Valore freno inviato al gioco")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()