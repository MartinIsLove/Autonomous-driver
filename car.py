import socket
import threading
import json

class Car:
    """Rappresenta un'auto completa con stato e controlli per TORCS"""
    
    def __init__(self, host='localhost', port=3001, broadcast_port=3002):
        # === STATO AUTO ===
        # Posizione e orientamento
        self.angle = 0.0           # Angolo rispetto alla direzione della pista
        self.trackPos = 0.0        # Posizione laterale (-1 = sinistra, +1 = destra)
        self.z = 0.0              # Altezza
        
        # Velocità
        self.speedX = 0.0         # Velocità longitudinale
        self.speedY = 0.0         # Velocità laterale  
        self.speedZ = 0.0         # Velocità verticale
        
        # Motore
        self.rpm = 0.0            # Giri motore
        self.gear = 0             # Marcia attuale
        
        # Sensori distanza (19 sensori di traccia)
        self.track = [200.0] * 19
        
        # Sensori avversari (36 sensori)
        self.opponents = [200.0] * 36
        
        # Sensori focus (5 sensori)
        self.focus = [200.0] * 5
        
        # Ruote
        self.wheelSpinVel = [0.0] * 4  # Velocità rotazione ruote
        
        # Stato gara
        self.fuel = 100.0         # Carburante
        self.damage = 0.0         # Danno auto
        self.distRaced = 0.0      # Distanza percorsa
        self.distFromStart = 0.0  # Distanza dal start
        self.curLapTime = 0.0     # Tempo giro corrente
        self.lastLapTime = 0.0    # Tempo ultimo giro
        self.racePos = 1          # Posizione in gara
        
        # === CONTROLLI AUTO ===
        self.throttle = 0.0      # Accelerazione (0.0 - 1.0)
        self.brake = 0.0      # Frenata (0.0 - 1.0)  
        self.steer = 0.0      # Sterzo (-1.0 - 1.0)
        self.control_gear = 1 # Marcia comandata (-1, 0, 1-6)
        self.clutch = 0.0     # Frizione (0.0 - 1.0)
        self.focus_control = 0 # Focus sensori
        self.meta = 0         # Meta comando
        
        # === CONTROLLO ===
        self.control = None
        
        # === CONNESSIONE TORCS ===
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        
        # Socket per broadcasting dati
        self.broadcast_port = broadcast_port
        self.broadcast_sock = None
        self.setup_broadcast()
    
    def connect(self):
        """Connette a TORCS"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0)  # Usa un timeout
            
            # Messaggio di inizializzazione
            init_msg = "SCR(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
            self.sock.sendto(init_msg.encode(), (self.host, self.port))
            
            self.connected = True
            print(f"Connesso a TORCS su {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"Errore connessione: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnette da TORCS"""
        if self.sock:
            self.sock.close()
        self.connected = False
        print("Disconnesso da TORCS")
    
    def setup_broadcast(self):
        """Setup socket per inviare dati a terminale separato"""
        try:
            self.broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Broadcasting su porta {self.broadcast_port}")
        except Exception as e:
            print(f"Errore setup broadcast: {e}")
    
    def parse_message(self, message):
        """Parser per i messaggi TORCS"""
        data_dict = {}
        message = message.strip()
        
        i = 0
        while i < len(message):
            if message[i] == '(':
                # Trova la fine della chiave
                j = i + 1
                while j < len(message) and message[j] != ' ':
                    j += 1
                
                key = message[i+1:j]
                
                # Trova i valori fino alla parentesi di chiusura
                values = []
                j += 1  # salta lo spazio
                start = j
                
                while j < len(message) and message[j] != ')':
                    if message[j] == ' ':
                        if start < j:
                            value = message[start:j]
                            try:
                                if '.' in value:
                                    values.append(float(value))
                                else:
                                    values.append(int(value))
                            except ValueError:
                                values.append(value)
                        start = j + 1
                    j += 1
                
                # Aggiungi l'ultimo valore
                if start < j:
                    value = message[start:j]
                    try:
                        if '.' in value:
                            values.append(float(value))
                        else:
                            values.append(int(value))
                    except ValueError:
                        values.append(value)
                
                # Se c'è un solo valore, non fare una lista
                if len(values) == 1:
                    data_dict[key] = values[0]
                else:
                    data_dict[key] = values
                
                i = j + 1
            else:
                i += 1
        
        return data_dict
    
    def update_state(self, data_dict):
        """Aggiorna lo stato dell'auto dai dati ricevuti"""
        # Aggiorna i valori dello stato se presenti
        if 'angle' in data_dict:
            self.angle = data_dict['angle']
        if 'trackPos' in data_dict:
            self.trackPos = data_dict['trackPos']
        if 'z' in data_dict:
            self.z = data_dict['z']
        if 'speedX' in data_dict:
            self.speedX = data_dict['speedX']
        if 'speedY' in data_dict:
            self.speedY = data_dict['speedY']
        if 'speedZ' in data_dict:
            self.speedZ = data_dict['speedZ']
        if 'rpm' in data_dict:
            self.rpm = data_dict['rpm']
        if 'gear' in data_dict:
            self.gear = data_dict['gear']
        if 'track' in data_dict:
            self.track = data_dict['track']
        if 'opponents' in data_dict:
            self.opponents = data_dict['opponents']
        if 'focus' in data_dict:
            self.focus = data_dict['focus']
        if 'wheelSpinVel' in data_dict:
            self.wheelSpinVel = data_dict['wheelSpinVel']
        if 'fuel' in data_dict:
            self.fuel = data_dict['fuel']
        if 'damage' in data_dict:
            self.damage = data_dict['damage']
        if 'distRaced' in data_dict:
            self.distRaced = data_dict['distRaced']
        if 'distFromStart' in data_dict:
            self.distFromStart = data_dict['distFromStart']
        if 'curLapTime' in data_dict:
            self.curLapTime = data_dict['curLapTime']
        if 'lastLapTime' in data_dict:
            self.lastLapTime = data_dict['lastLapTime']
        if 'racePos' in data_dict:
            self.racePos = data_dict['racePos']
    
    def broadcast_data(self, data_dict):
        """Invia dati al terminale di monitoraggio"""
        if self.broadcast_sock:
            try:
                message = json.dumps(data_dict, indent=2)
                self.broadcast_sock.sendto(message.encode(), ('localhost', self.broadcast_port))
            except Exception as e:
                print(f"Errore broadcast: {e}")
    
    def receive_data(self):
        """Riceve e processa i dati da TORCS"""
        if not self.connected or not self.sock:
            return None
        
        try:
            data, addr = self.sock.recvfrom(1024)
            message = data.decode()

            # Handle special messages
            if '***RESTART***' in message:
                print("🔄 Restart signal received.")
                return {'special': 'restart'}
            
            if '***SHUTDOWN***' in message:
                print("🛑 Shutdown signal received.")
                return {'special': 'shutdown'}

            data_dict = self.parse_message(message)
            self.update_state(data_dict)
            
            # Broadcast dei dati
            self.broadcast_data(data_dict)
            
            return data_dict
        except socket.timeout:
            return None # Non è un errore, succede
        except Exception as e:
            print(f"❌ Errore ricezione dati: {e}")
            return None
    
    def send_controls(self):
        """Invia i comandi di controllo a TORCS"""
        if not self.connected or not self.sock:
            return False
        
        try:
            control_msg = self.to_torcs_string()
            self.sock.sendto(control_msg.encode(), (self.host, self.port))
            return True
        except Exception as e:
            print(f"Errore invio comandi: {e}")
            return False
    
    def __str__(self):
        """Rappresentazione stringa completa"""
        return (f"Car(speed={self.speedX:.1f}, angle={self.angle:.3f}, "
                f"trackPos={self.trackPos:.3f}, gear={self.gear}, rpm={self.rpm:.0f}, "
                f"throttle={self.throttle:.2f}, brake={self.brake:.2f}, steer={self.steer:.2f})")
    
    def to_torcs_string(self):
        """Converte i controlli in stringa formato TORCS"""
        return (f"(accel {self.throttle})(brake {self.brake})"
                f"(gear {self.control_gear})(steer {self.steer})"
                f"(clutch {self.clutch})(focus {self.focus_control})(meta {self.meta})")
    
    def validate_and_clip_controls(self):
        """Valida e limita i valori di controllo nei range corretti"""
        self.throttle = max(0.0, min(1.0, self.throttle))
        self.brake = max(0.0, min(1.0, self.brake))
        self.steer = max(-1.0, min(1.0, self.steer))
        self.control_gear = max(-1, min(6, int(self.control_gear)))
        self.clutch = max(0.0, min(1.0, self.clutch))
        return self
    
    def set_controls(self, throttle=None, brake=None, steer=None, gear=None, clutch=None):
        """Imposta i controlli dell'auto"""
        if throttle is not None:
            self.throttle = throttle
        if brake is not None:
            self.brake = brake
        if steer is not None:
            self.steer = steer
        if gear is not None:
            self.control_gear = gear
        if clutch is not None:
            self.clutch = clutch
        
        return self.validate_and_clip_controls()
    
    def get_state_dict(self):
        """Restituisce lo stato corrente dell'auto come dizionario."""
        return {
            'angle': self.angle,
            'trackPos': self.trackPos,
            'z': self.z,
            'speedX': self.speedX,
            'speedY': self.speedY,
            'speedZ': self.speedZ,
            'rpm': self.rpm,
            'gear': self.gear,
            'track': self.track,
            'opponents': self.opponents,
            'focus': self.focus,
            'wheelSpinVel': self.wheelSpinVel,
            'fuel': self.fuel,
            'damage': self.damage,
            'distRaced': self.distRaced,
            'distFromStart': self.distFromStart,
            'curLapTime': self.curLapTime,
            'lastLapTime': self.lastLapTime,
            'racePos': self.racePos
        }
    
    def set_control(self, control):
        """Imposta la strategia di controllo."""
        self.control = control
