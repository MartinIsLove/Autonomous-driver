from .base_control import BaseControl
from typing import Dict, Any
import pickle
import numpy as np
import warnings

# Silenzia i warning di sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class AIControl(BaseControl):
    '''
    Driver per TORCS che utilizza il modello di machine learning
    '''

    def __init__(self, model_path='torcs_optimal_model.pkl'):
        self.model_path = model_path
        self.prev_gear = 3
        self.models = {}
        self.scalers = {}
        self.feature_sets = {}
        self.optimal_models = {}
        
        # Carica il modello
        self.load_model()

    def load_model(self):
        """Carica il modello di ML ottimale"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_sets = model_data['feature_sets']
            self.optimal_models = model_data['optimal_models']
            
            print(f"Modello caricato da: {self.model_path}")
            print(f"Modelli ottimali:")
            for target, model_type in self.optimal_models.items():
                print(f"   {target}: {model_type.upper()}")
            
        except Exception as e:
            print(f"Errore caricamento modello: {e}")
            raise

    def get_sensor_dict(self, car_state_obj):
        """Estrae i dati dei sensori dall'oggetto Car e li mette in un dizionario."""
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
        }
        for i, val in enumerate(car_state_obj.track):
            sensor_dict[f'track_{i}'] = val
        for i, val in enumerate(car_state_obj.opponents):
            sensor_dict[f'opponents_{i}'] = val
        return sensor_dict

    def predict_controls(self, sensor_data):
        """Predice i controlli usando il modello ottimale"""
        
        predictions = {}
        
        for target_name in ['steer', 'throttle', 'brake', 'gear']:
            try:
                features = self.feature_sets[target_name]
                
                # Usa sempre array numpy per evitare warning
                X_input = np.array([[sensor_data.get(f, 0.0) for f in features]])
                
                model_type = self.optimal_models[target_name]
                
                if model_type == 'nn':
                    if target_name in self.scalers:
                        X_input_scaled = self.scalers[target_name].transform(X_input)
                        pred = self.models[target_name].predict(X_input_scaled)[0]
                    else:
                        pred = self.models[target_name].predict(X_input)[0]
                else:
                    pred = self.models[target_name].predict(X_input)[0]
                
                if target_name == 'steer':
                    pred = np.clip(pred, -1.0, 1.0)
                elif target_name in ['throttle', 'brake']:
                    pred = np.clip(pred, 0.0, 1.0)
                elif target_name == 'gear':
                    pred = int(max(-1, min(6, round(pred))))
                
                predictions[target_name] = pred
                
            except Exception as e:
                print(f"Errore predizione {target_name}: {e}")
                fallback_values = {
                    'steer': 0.0,
                    'throttle': 0.3,
                    'brake': 0.0,
                    'gear': 3
                }
                predictions[target_name] = fallback_values[target_name]
        
        return predictions
    
    def safety_adjustments(self, controls, car_state_obj):
        """Aggiustamenti di sicurezza basati sullo stato dell'auto"""
        
        if car_state_obj.speedX < 5 and controls['gear'] == 0:
            controls['gear'] = 1
            
        if car_state_obj.speedX < -5:
            controls['gear'] = -1
            
        if controls['throttle'] > 0.1 and controls['brake'] > 0.1:
            if controls['throttle'] > controls['brake']:
                controls['brake'] = 0.0
            else:
                controls['throttle'] = 0.0
        
        if abs(car_state_obj.angle) > 1.0:
            controls['throttle'] *= 0.5
            controls['brake'] = max(controls['brake'], 0.3)
            
        gear_diff = abs(controls['gear'] - self.prev_gear)
        if gear_diff > 2:
            if controls['gear'] > self.prev_gear:
                controls['gear'] = self.prev_gear + 1
            else:
                controls['gear'] = self.prev_gear - 1
        
        self.prev_gear = controls['gear']
        
        return controls
    
    def get_actions(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementa la logica dell'AI per decidere le azioni
        in base allo stato del gioco/simulazione.
        """
        car_state_obj = kwargs.get('game_state')
        if not car_state_obj:
            return {"steer": 0.0, "throttle": 0.3, "brake": 0.0, "gear": 1}

        sensor_data = self.get_sensor_dict(car_state_obj)
        
        # Predici controlli
        predictions = self.predict_controls(sensor_data)
        
        # Aggiustamenti di sicurezza
        # final_controls = self.safety_adjustments(predictions.copy(), car_state_obj)
        final_controls = predictions
        
        return {
            "steer": float(final_controls['steer']),
            "throttle": float(final_controls['throttle']),
            "brake": float(final_controls['brake']),
            "gear": int(final_controls['gear'])
        }
