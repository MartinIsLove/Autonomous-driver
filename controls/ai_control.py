from .base_control import BaseControl
from typing import Dict, Any
import pickle
import numpy as np
import warnings
import threading
import queue
import time

# Silenzia i warning di sklearn
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
        self.feature_sets = {}
        self.optimal_models = {}
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.ai_thread = None
        self.last_controls = {"steer": 0.0, "throttle": 0.3, "brake": 0.0, "gear": 1}
        
        # Carica il modello
        self.load_model()
        
        # Avvia il thread AI
        self.start_ai_thread()

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
        """Worker thread che elabora le predizioni AI"""
        while not self.stop_event.is_set():
            try:
                # Prendi i dati più recenti dalla queue (non bloccante)
                car_state_obj = None
                try:
                    car_state_obj = self.input_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.001)  # Breve pausa per non sovraccaricare la CPU
                    continue
                
                if car_state_obj is None:
                    continue
                
                # Elabora i controlli
                sensor_data = self.get_sensor_dict(car_state_obj)
                predictions = self.predict_controls(sensor_data)
                final_controls = predictions  # o self.safety_adjustments(predictions.copy(), car_state_obj)
                
                # Metti il risultato nella queue di output (sostituisce se piena)
                try:
                    self.output_queue.put_nowait(final_controls)
                except queue.Full:
                    # Rimuovi il vecchio risultato e metti quello nuovo
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(final_controls)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                print(f"Errore nel thread AI: {e}")
                time.sleep(0.1)

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
                    'throttle': 0.0,
                    'brake': 0.0,
                    'gear': 0
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
        in base allo stato del gioco/simulazione (thread-safe).
        """
        car_state_obj = kwargs.get('game_state')
        if not car_state_obj:
            return self.last_controls.copy()

        # Invia i dati al thread AI (non bloccante)
        try:
            self.input_queue.put_nowait(car_state_obj)
        except queue.Full:
            # Sostituisci i dati vecchi con quelli nuovi
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait(car_state_obj)
            except queue.Empty:
                pass
        
        # Prendi il risultato più recente se disponibile
        try:
            new_controls = self.output_queue.get_nowait()
            self.last_controls = {
                "steer": float(new_controls['steer']),
                "throttle": float(new_controls['throttle']),
                "brake": float(new_controls['brake']),
                "gear": int(new_controls['gear'])
            }
        except queue.Empty:
            # Usa i controlli precedenti se non ci sono nuovi risultati
            pass
        
        return self.last_controls.copy()

    def __del__(self):
        """Cleanup quando l'oggetto viene distrutto"""
        self.stop_ai_thread()