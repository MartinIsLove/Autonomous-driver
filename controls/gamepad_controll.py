import pygame
import numpy as np
from .base_control import BaseControl

class DualShockControl(BaseControl):
    """Gestisce i controlli da controller DualShock PS4"""

    def __init__(self):
        """Inizializza pygame, il joystick e lo stato dei controlli"""
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise IOError("Nessun controller trovato")
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Controller inizializzato: {self.joystick.get_name()}")

        self.controls = {'steer': 0.0, 'throttle': 0.0, 'brake': 0.0, 'gear': 0}
        self.prev_gear_up = False
        self.prev_gear_down = False
        
        # Zona morta
        self.deadzone = 0.1

    def apply_deadzone(self, value, deadzone=None):
        """Applica la zona morta a un valore analogico"""
        if deadzone is None:
            deadzone = self.deadzone
            
        if abs(value) < deadzone:
            return 0.0
        else:
            sign = 1 if value > 0 else -1
            return sign * (abs(value) - deadzone) / (1.0 - deadzone)

    def get_actions(self, *args, **kwargs):
        """
        Legge l'input dal controller
        """
        pygame.event.pump()
        
        # Sterzo (stick sinistro - X)
        raw_steer = self.joystick.get_axis(0)
        steer_with_deadzone = self.apply_deadzone(-raw_steer)
        self.controls['steer'] = round(max(-1.0, min(1.0, steer_with_deadzone)), 3)
        
        # Acceleratore (R2)
        throttle_axis = self.joystick.get_axis(5)  # R2 su PS4
        raw_throttle = max(0.0, (throttle_axis + 1) / 2)
        throttle_with_deadzone = self.apply_deadzone(raw_throttle, deadzone=0.05)  # 5% zona morta
        self.controls['throttle'] = round(max(0.0, min(1.0, throttle_with_deadzone)), 3)
        
        # Freno (L2)
        brake_axis = self.joystick.get_axis(4)  # L2 su PS4
        raw_brake = max(0.0, (brake_axis + 1) / 2)
        brake_with_deadzone = self.apply_deadzone(raw_brake, deadzone=0.05)  # 5% zona morta
        if brake_with_deadzone > 0:
            self.controls['brake'] = round(max(0.0, min(1.0, brake_with_deadzone)), 3)
        else:
            self.controls['brake'] = 0.0

        # Cambio (X per marcia su, Quadrato per marcia giù)
        gear_up = self.joystick.get_button(0) 
        gear_down = self.joystick.get_button(2)
        
        if gear_up and not self.prev_gear_up and self.controls['gear'] < 6:
            self.controls['gear'] += 1
        if gear_down and not self.prev_gear_down and self.controls['gear'] > -1:
            self.controls['gear'] -= 1
            
        self.prev_gear_up = gear_up
        self.prev_gear_down = gear_down
        
        return self.controls.copy()