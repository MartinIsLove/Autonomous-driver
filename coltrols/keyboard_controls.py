import pynput
from pynput import keyboard

key_actions = {
    'accelerate': False,
    'brake': False,
    'steer_left': False,
    'steer_right': False
}

def update_controls(controls):
    # Throttle e brake
    if key_actions['accelerate'] and not key_actions['brake'] and controls['throttle'] <= 0.8:
        controls['throttle'] += 0.2
        controls['brake'] = 0.0
    elif key_actions['brake'] and not key_actions['accelerate'] and controls['brake'] <= 0.9:
        controls['throttle'] = 0.0
        controls['brake'] += 0.1
    elif not key_actions['accelerate'] and not key_actions['brake']:
        controls['throttle'] = 0.0
        controls['brake'] = 0.0
    
    # Sterzo
    if key_actions['steer_left'] and not key_actions['steer_right'] and controls['steer'] <= 0.8:
        controls['steer'] += 0.2
    elif key_actions['steer_right'] and not key_actions['steer_left'] and controls['steer'] >= -0.8:
        controls['steer'] -= 0.2
    elif not key_actions['steer_left'] and not key_actions['steer_right']:
        controls['steer'] = 0.0
    
    return controls

def on_press(key, controls):
    try:
        if hasattr(key, 'char'):
            if key.char == 'w':
                key_actions['accelerate'] = True
            elif key.char == 's':
                key_actions['brake'] = True
            elif key.char == 'a':
                key_actions['steer_left'] = True
            elif key.char == 'd':
                key_actions['steer_right'] = True
            elif key.char == 'q':
                controls['gear'] = max(-1, controls['gear'] - 1)
            elif key.char == 'e':
                controls['gear'] = min(6, controls['gear'] + 1)
            elif key.char == 'r':
                controls['throttle'] = 0.0
                controls['brake'] = 0.0
                controls['steer'] = 0.0
                key_actions['accelerate'] = False
                key_actions['brake'] = False
                key_actions['steer_left'] = False
                key_actions['steer_right'] = False
            controls = update_controls(controls)
    except AttributeError:
        pass
    return controls

def on_release(key, controls):
    try:
        if hasattr(key, 'char'):
            if key.char == 'w':
                key_actions['accelerate'] = False
            elif key.char == 's':
                key_actions['brake'] = False
            elif key.char == 'a':
                key_actions['steer_left'] = False
            elif key.char == 'd':
                key_actions['steer_right'] = False
            controls = update_controls(controls)
    except AttributeError:
        pass
    return controls

def get_keyboard_controls(controls):
    listener = keyboard.Listener(
        on_press=lambda key: on_press(key, controls),
        on_release=lambda key: on_release(key, controls)
    )
    listener.start()
    return controls
