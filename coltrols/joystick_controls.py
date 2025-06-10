import math
import pygame
import matplotlib.pyplot as plt
import numpy as np

def get_joystick_controls(controls, joystick, prev_gear_up, prev_gear_down):
    pygame.event.pump()
    raw_steer = joystick.get_axis(0) * -1
    steer = max(-1.0, min(1.0, raw_steer * 1.5))
    throttle = (1 - joystick.get_axis(1)) / 2
    raw_brake = (1 - joystick.get_axis(2)) / 2
    brake = 1 - np.log1p(180 * (1 - raw_brake)) / np.log1p(100)

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
    pedal = np.linspace(0, 1, 100)
    # Curva logaritmica invertita: parte da 0,0 e arriva a 1,1 con la gobba in basso
    brake_modulated = 1 - np.log1p(100 * (1 - pedal)) / np.log1p(100)
    
    plt.figure(figsize=(7, 5))
    plt.plot(pedal, brake_modulated, label="Curva freno (logaritmica invertita)")
    plt.plot(pedal, pedal, '--', label="Lineare (nessuna modulazione)")
    plt.title("Risposta del freno: pedale reale vs valore inviato al gioco")
    plt.xlabel("Pedale freno reale (non premuto → premuto)")
    plt.ylabel("Valore freno inviato al gioco")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()