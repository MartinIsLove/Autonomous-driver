import pygame
import numpy as np
from .base_control import BaseControl

class SteeringWheelControl(BaseControl):
    """Gestisce i controlli da volante."""

    def __init__(self):
        """Inizializza pygame, il joystick e lo stato dei controlli."""
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise IOError("Nessun volante/joystick trovato.")
            
        self.joystick = pygame.joystick.Joystick(1)
        self.joystick.init()
        
        print(f"Volante inizializzato: {self.joystick.get_name()}")

        self.controls = {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0, 'gear': 0}
        self.prev_gear_up = False
        self.prev_gear_down = False

    def get_actions(self, *args, **kwargs):
        """
        Legge l'input dal volante, calcola le azioni e le restituisce.
        """
        pygame.event.pump()
        
        # Sterzo
        raw_steer = self.joystick.get_axis(0) * -1
        self.controls['steer'] = max(-1.0, min(1.0, raw_steer * 1.5))
        
        # Acceleratore
        throttle_axis = self.joystick.get_axis(1)
        if throttle_axis > 0.95:  # Deadzone
            throttle_axis = 1.0
        self.controls['throttle'] = (1 - throttle_axis) / 2
        
        # Freno (con curva logaritmica)
        brake_axis = self.joystick.get_axis(2)
        if brake_axis > 0.95:  # Deadzone 
            brake_axis = 1.0
        raw_brake = (1 - brake_axis) / 2
        self.controls['brake'] = 1 - np.log(1 + 180 * (1 - raw_brake)) / np.log(1 + 180)

        # Cambio
        gear_up = self.joystick.get_button(4)
        gear_down = self.joystick.get_button(5)
        
        if gear_up and not self.prev_gear_up and self.controls['gear'] < 6:
            self.controls['gear'] += 1
        if gear_down and not self.prev_gear_down and self.controls['gear'] > -1:
            self.controls['gear'] -= 1
            
        self.prev_gear_up = gear_up
        self.prev_gear_down = gear_down
        
        return self.controls.copy()

