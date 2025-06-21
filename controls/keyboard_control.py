from .base_control import BaseControl
import keyboard

class KeyboardControl(BaseControl):
    """Gestisce i controlli da tastiera utilizzando la libreria 'keyboard'."""

    def __init__(self):
        self.controls = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0,
            'gear': 1,
            'clutch': 0.0
        }
        # Mappatura per i cambi di marcia per evitare cambiate multiple
        self.gear_keys_pressed = {
            'q': False,
            'e': False
        }

    def get_actions(self, *args, **kwargs):
        """
        Restituisce le azioni di controllo basate sullo stato attuale dei tasti.
        """
        # Accelerazione e freno
        if keyboard.is_pressed('w'):
            self.controls['throttle'] = min(1.0, self.controls['throttle'] + 0.2)
            self.controls['brake'] = 0.0
        elif keyboard.is_pressed('s'):
            self.controls['throttle'] = 0.0
            self.controls['brake'] = min(1.0, self.controls['brake'] + 0.1)
        else:
            self.controls['throttle'] *= 0 # Decelerazione graduale
            self.controls['brake'] *= 0
        # Sterzo
        if keyboard.is_pressed('d'):
            self.controls['steer'] = max(-1.0, self.controls['steer'] - 0.2)
        elif keyboard.is_pressed('a'):
            self.controls['steer'] = min(1.0, self.controls['steer'] + 0.2)
        else:
            self.controls['steer'] *= 0  # Ritorno graduale al centro

        # Cambio marcia (con controllo per pressione singola)
        if keyboard.is_pressed('q') and not self.gear_keys_pressed['q']:
            self.controls['gear'] = max(-1, self.controls['gear'] - 1)
            self.gear_keys_pressed['q'] = True
        elif not keyboard.is_pressed('q'):
            self.gear_keys_pressed['q'] = False

        if keyboard.is_pressed('e') and not self.gear_keys_pressed['e']:
            self.controls['gear'] = min(6, self.controls['gear'] + 1)
            self.gear_keys_pressed['e'] = True
        elif not keyboard.is_pressed('e'):
            self.gear_keys_pressed['e'] = False

        return self.controls
