from .base_control import BaseControl
from typing import Dict, Any
import pickle
import numpy as np
import warnings
import threading
import time

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class AIControl(BaseControl):
    '''
    Driver per TORCS che utilizza il modello di machine learning in un thread separato
    '''

    def __init__(self, model_path='torcs_optimal_model.pkl'):
        self.model_path = model_path
        self.prev_gear = 3
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_sets = {}
        self.optimal_models = {}
        
        self.stop_event = threading.Event()
        self.ai_thread = None
        self.current_state = None
        self.current_controls = {"steer": 0.0, "throttle": 0.0, "brake": 0.0, "gear": 0}
        self.state_lock = threading.Lock()
        
        self.load_model()
        
        self.start_ai_thread()

    def load_model(self):
        """Carica il modello di ML ottimale"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.encoders = model_data.get('encoders', {})
            self.feature_sets = model_data['feature_sets']
            self.optimal_models = model_data['optimal_models']
            
            print(f"Modello caricato da: {self.model_path}")
            print(f"Modelli ottimali:")
            for target, model_type in self.optimal_models.items():
                print(f"   {target}: {model_type.upper()}")
            
        except Exception as e:
            print(f"Errore caricamento modello: {e}")
            raise

    def start_ai_thread(self):
        """Avvia il thread per l'elaborazione AI"""
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()
        print("Thread AI avviato")

    def stop_ai_thread(self):
        """Ferma il thread AI"""
        self.stop_event.set()
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        print("Thread AI fermato")

    def _ai_worker(self):
        """Worker per il thread AI che elabora i dati e predice i controlli"""
        while not self.stop_event.is_set():
            try:
                with self.state_lock:
                    car_state = self.current_state
                
                if car_state is None:
                    time.sleep(0.001)
                    continue
                
                sensor_data = self.get_sensor_dict(car_state)
                predictions = self.predict_controls(sensor_data)
                
                predictions = self.safety_adjustments(predictions, car_state)
                
                with self.state_lock:
                    self.current_controls = predictions
                
            except Exception as e:
                print(f"Errore nel thread AI: {e}")
                time.sleep(0.1)

    def get_sensor_dict(self, car_state_obj):
        """Estrae i dati dei sensori dall'oggetto Car e li mette in un dizionario"""
        sensor_dict = {
            'angle': car_state_obj.angle,
            'trackPos': car_state_obj.trackPos,
            'speedX': car_state_obj.speedX,
            'speedY': car_state_obj.speedY,
            'speedZ': car_state_obj.speedZ,
            'rpm': car_state_obj.rpm,
            'gear': car_state_obj.gear,
            'fuel': car_state_obj.fuel,
            'damage': car_state_obj.damage,
            'distRaced': car_state_obj.distRaced,
            'distFromStart': car_state_obj.distFromStart,
            'curLapTime': car_state_obj.curLapTime,
            'lastLapTime': car_state_obj.lastLapTime,
            'throttle': getattr(car_state_obj, 'throttle', 0.0),
            'brake': getattr(car_state_obj, 'brake', 0.0),
            'steer': getattr(car_state_obj, 'steer', 0.0),
        }
        for i, val in enumerate(car_state_obj.track):
            sensor_dict[f'track_{i}'] = val
        return sensor_dict

    def predict_controls(self, sensor_data):
        """Predice i controlli usando il modello ottimale con remapping corretto"""
        
        predictions = {}
        
        # STEER
        try:
            features_steer = self.feature_sets['steer']
            X_input_steer = np.array([[sensor_data.get(f, 0.0) for f in features_steer]])
            
            if 'steer' in self.scalers:
                X_input_steer = self.scalers['steer'].transform(X_input_steer)
            
            pred_steer = self.models['steer'].predict(X_input_steer)[0]
            predictions['steer'] = np.clip(pred_steer, -1.0, 1.0)
            
        except Exception as e:
            print(f"Errore predizione steer: {e}")
            predictions['steer'] = 0.0
        
        # THROTTLE
        try:
            features_throttle = self.feature_sets['throttle']
            X_input_throttle = np.array([[sensor_data.get(f, 0.0) for f in features_throttle]])
            
            if 'throttle' in self.scalers:
                X_input_throttle = self.scalers['throttle'].transform(X_input_throttle)
            
            pred_throttle = self.models['throttle'].predict(X_input_throttle)[0]
            predictions['throttle'] = np.clip(pred_throttle, 0.0, 1.0)
            
        except Exception as e:
            print(f"Errore predizione throttle: {e}")
            predictions['throttle'] = 0.0
        
        # BRAKE
        try:
            features_brake = self.feature_sets['brake']
            X_input_brake = np.array([[sensor_data.get(f, 0.0) for f in features_brake]])
            
            if 'brake' in self.scalers:
                X_input_brake = self.scalers['brake'].transform(X_input_brake)
            
            pred_brake = self.models['brake'].predict(X_input_brake)[0]
            predictions['brake'] = np.clip(pred_brake, 0.0, 1.0)
            
        except Exception as e:
            print(f"Errore predizione brake: {e}")
            predictions['brake'] = 0.0
        
        # GEAR
        try:
            features_gear = self.feature_sets['gear']
            X_input_gear = np.array([[sensor_data.get(f, 0.0) for f in features_gear]])
            
            if 'gear' in self.scalers:
                X_input_gear = self.scalers['gear'].transform(X_input_gear)
            
            pred_gear_encoded = self.models['gear'].predict(X_input_gear)[0]
            pred_gear = self.encoders['gear'].inverse_transform([pred_gear_encoded])[0]
            predictions['gear'] = int(pred_gear)
            
        except Exception as e:
            print(f"Errore predizione gear: {e}")
            predictions['gear'] = 0
        
        return predictions
    
    def safety_adjustments(self, controls, car_state):
        """Aggiustamenti di sicurezza"""
        
        max_speed = 100
        if car_state.speedX > max_speed:
            controls['gear'] = 2
        
        return controls
    
    def get_actions(self, *args, **kwargs) -> Dict[str, Any]:
        """Versione semplificata senza queue"""
        car_state_obj = kwargs.get('game_state')
        if not car_state_obj:
            with self.state_lock:
                return self.current_controls.copy()

        with self.state_lock:
            self.current_state = car_state_obj
            return self.current_controls.copy()

    def __del__(self):
        """Cleanup quando l'oggetto viene distrutto"""
        self.stop_ai_thread()