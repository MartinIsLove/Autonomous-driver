import tkinter as tk
from tkinter import font
import socket
import json
import threading
import time
import queue
import math

class TorcsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🏎️ TORCS Data Monitor (JSON Version)")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        self.port = 3002
        self.sock = None
        self.running = False
        self.data_count = 0
        self.start_time = time.time()
        self.data_queue = queue.Queue()
        
        self.title_font = font.Font(family="Arial", size=12, weight="bold")
        self.data_font = font.Font(family="Consolas", size=10)
        self.small_font = font.Font(family="Arial", size=9)
        
        self.setup_gui()
        self.setup_socket()
        self.start_monitoring()
        
    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = tk.Label(main_frame, text="🏎️ TORCS Data Monitor (JSON)", font=("Arial", 16, "bold"), fg='#00ccff', bg='#1e1e1e')
        title_label.pack(pady=(0, 10))
        
        columns_frame = tk.Frame(main_frame, bg='#1e1e1e')
        columns_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(columns_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = tk.Frame(columns_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.create_main_info_section(left_frame)
        self.create_sensors_section(left_frame)
        self.create_position_section(right_frame)
        self.create_wheels_section(right_frame)
        self.create_opponents_section(right_frame)
        self.create_system_section(right_frame)
        
    def create_section_frame(self, parent, title):
        section_frame = tk.LabelFrame(parent, text=title, font=self.title_font, fg='#00ccff', bg='#2d2d2d', relief=tk.GROOVE, bd=2)
        section_frame.pack(fill=tk.X, padx=5, pady=5, ipady=2)
        return section_frame
        
    def create_main_info_section(self, parent):
        frame = self.create_section_frame(parent, "STATO AUTO")
        self.main_labels = self._create_labels(frame, [
            ("🚀 Velocità:", "speed_kmh", "km/h"), ("📏 speedX", "speedX_ms", ""),
            ("⚙️ Marcia:", "gear", ""), ("📏 speedY:", "speedY", ""), 
            ("🔧 RPM:", "rpm", "giri/min"),  ("📍 Pos. Pista:", "trackPos", ""),
            ("💥 Danno:", "damage", ""), ("🧭 Angolo:", "angle", "rad"),
            ("⛽ Carburante:", "fuel", "%")
        ], '#00ff00')
    
    def create_sensors_section(self, parent):
        frame = self.create_section_frame(parent, "🛣️ SENSORI PISTA")
        self.sensor_canvas = tk.Canvas(frame, width=400, height=400, bg='#2d2d2d', highlightthickness=0)
        self.sensor_canvas.pack(pady=5)
        
        self.sensor_lines = []
        self.sensor_angles = [-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90]
        center_x, center_y = 200, 200
        max_len = 190

        # Car representation
        self.sensor_canvas.create_rectangle(center_x - 8, center_y - 15, center_x + 8, center_y + 15, fill='#00ccff')

        for angle in self.sensor_angles:
            rad_angle = math.radians(angle - 90)
            end_x = center_x + max_len * math.cos(rad_angle)
            end_y = center_y + max_len * math.sin(rad_angle)
            line = self.sensor_canvas.create_line(center_x, center_y, end_x, end_y, fill='green', width=2)
            self.sensor_lines.append(line)
    
    def create_position_section(self, parent):
        frame = self.create_section_frame(parent, "📏 POSIZIONE & TEMPI")
        self.position_labels = self._create_labels(frame, [
            ("🏁 Dist. Start:", "distFromStart", "m"), ("🛣️ Dist. Percorsa:", "distRaced", "m"),
            ("📐 Altezza Z:", "z", "m"), ("⏱️ Tempo Giro:", "curLapTime", "s"),
            ("🕐 Ultimo Giro:", "lastLapTime", "s"), ("🏁 Posizione:", "racePos", ""),
        ], '#ffff00')
    
    def create_wheels_section(self, parent):
        frame = self.create_section_frame(parent, "🛞 RUOTE")
        self.wheel_labels = {}
        for i, pos in enumerate(["🔴 Ant. Sin.", "🟢 Ant. Des.", "🔵 Post. Sin.", "🟡 Post. Des."]):
            tk.Label(frame, text=pos, font=self.small_font, fg='white', bg='#2d2d2d').grid(row=i, column=0, sticky='w', padx=5, pady=1)
            speed_label = tk.Label(frame, text="--- rad/s", font=self.data_font, fg='#ff00ff', bg='#2d2d2d', width=12, anchor='w')
            speed_label.grid(row=i, column=1, padx=5, pady=1)
            self.wheel_labels[i] = speed_label
    
    def create_opponents_section(self, parent):
        frame = self.create_section_frame(parent, "🏁 AVVERSARI")
        self.opponent_labels = self._create_labels(frame, [
            ("🔴 Molto vicini (<30m):", "very_close", ""),
            ("🟡 Vicini (<100m):", "close", ""),
            ("🟢 Totali rilevati:", "total", ""),
        ], '#ff6666', single_col=True)
    
    def create_system_section(self, parent):
        frame = self.create_section_frame(parent, "📊 SISTEMA")
        self.system_labels = self._create_labels(frame, [
            ("📦 Pacchetti:", "packets", ""), ("⏱️ Tempo attivo:", "uptime", ""),
            ("📡 FPS:", "fps", ""), ("🕒 Ultimo agg.:", "last_update", ""),
        ], '#ffffff', single_col=True)
    
    def _create_labels(self, parent, items, color, single_col=False):
        labels = {}
        for i, (label, key, unit) in enumerate(items):
            row, col = (i, 0) if single_col else divmod(i, 2)
            tk.Label(parent, text=label, font=self.small_font, fg='white', bg='#2d2d2d').grid(row=row, column=col*3, sticky='w', padx=5, pady=2)
            value_label = tk.Label(parent, text="---", font=self.data_font, fg=color, bg='#2d2d2d', width=15 if single_col else 8, anchor='w')
            value_label.grid(row=row, column=col*3+1, padx=5, pady=2)
            if unit:
                tk.Label(parent, text=unit, font=self.small_font, fg='#888888', bg='#2d2d2d').grid(row=row, column=col*3+2, sticky='w', pady=2)
            labels[key] = value_label
        return labels
    
    def setup_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('localhost', self.port))
            self.sock.settimeout(1.0)
            print(f"GUI Monitor (JSON) in ascolto sulla porta {self.port}")
        except Exception as e:
            print(f"Errore setup socket: {e}")
    
    def safe_update_label(self, label, text, color=None):
        if label and label.winfo_exists():
            label.config(text=str(text))
            if color:
                label.config(fg=color)
    
    def update_gui(self, data_dict):
        """Aggiorna l'interfaccia con i nuovi dati."""
        speed_kmh = data_dict.get('speedX', 0.0)
        speed_ms = speed_kmh / 3.6 
        
        self.safe_update_label(self.main_labels['speed_kmh'], f"{speed_kmh:.1f}")
        self.safe_update_label(self.main_labels['speedX_ms'], f"{speed_ms:.3f}")
        
        self.safe_update_label(self.main_labels['speedY'], f"{data_dict.get('speedY', 0.0):.3f}")
        self.safe_update_label(self.main_labels['angle'], f"{data_dict.get('angle', 0.0):.4f}")
        self.safe_update_label(self.main_labels['trackPos'], f"{data_dict.get('trackPos', 0.0):.3f}")
        self.safe_update_label(self.main_labels['gear'], f"{int(data_dict.get('gear', 0))}")
        self.safe_update_label(self.main_labels['rpm'], f"{data_dict.get('rpm', 0.0):.0f}")
        self.safe_update_label(self.main_labels['fuel'], f"{data_dict.get('fuel', 0.0):.1f}")

        self.safe_update_label(self.main_labels['damage'], f"{int(data_dict.get('damage', 0))}")
        
        self.safe_update_label(self.position_labels['distFromStart'], f"{data_dict.get('distFromStart', 0.0):.1f}")
        self.safe_update_label(self.position_labels['distRaced'], f"{data_dict.get('distRaced', 0.0):.1f}")
        self.safe_update_label(self.position_labels['z'], f"{data_dict.get('z', 0.0):.3f}")
        self.safe_update_label(self.position_labels['curLapTime'], f"{data_dict.get('curLapTime', 0.0):.2f}")
        self.safe_update_label(self.position_labels['lastLapTime'], f"{data_dict.get('lastLapTime', 0.0):.2f}")
        self.safe_update_label(self.position_labels['racePos'], f"{int(data_dict.get('racePos', 0))}")
        
        track_data = data_dict.get('track', [200.0] * 19)
        if len(track_data) >= 19 and hasattr(self, 'sensor_canvas'):
            center_x, center_y = 200, 200
            max_len = 190
            for i, distance in enumerate(track_data):
                if i < len(self.sensor_lines):
                    line = self.sensor_lines[i]
                    angle = self.sensor_angles[i]
                    rad_angle = math.radians(angle - 90)
                    
                    line_len = min(distance, 200.0) / 200.0 * max_len
                    
                    end_x = center_x + line_len * math.cos(rad_angle)
                    end_y = center_y + line_len * math.sin(rad_angle)
                    
                    self.sensor_canvas.coords(line, center_x, center_y, end_x, end_y)
                    
                    color = '#ff0000' if distance < 20 else '#ffff00' if distance < 50 else '#00ff00'
                    self.sensor_canvas.itemconfig(line, fill=color)
        
        wheels = data_dict.get('wheelSpinVel', [0.0] * 4)
        for i, speed in enumerate(wheels):
            self.safe_update_label(self.wheel_labels[i], f"{speed:.3f} rad/s")
        
        opponents = data_dict.get('opponents', [])
        self.safe_update_label(self.opponent_labels['very_close'], sum(1 for d in opponents if d < 30))
        self.safe_update_label(self.opponent_labels['close'], sum(1 for d in opponents if d < 100))
        self.safe_update_label(self.opponent_labels['total'], sum(1 for d in opponents if d < 200))
    
    def monitor_thread(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                message = data.decode('utf-8')
                data_dict = json.loads(message)
                self.data_queue.put(data_dict)
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print(f"Errore decodifica JSON: {message[:150]}...")
            except Exception as e:
                if self.running: print(f"Errore monitor: {e}")
    
    def process_queue(self):
        try:
            while not self.data_queue.empty():
                data_dict = self.data_queue.get_nowait()
                self.data_count += 1
                self.update_gui(data_dict)
        except queue.Empty:
            pass
        
        elapsed = time.time() - self.start_time
        fps = self.data_count / elapsed if elapsed > 0 else 0
        self.safe_update_label(self.system_labels['packets'], str(self.data_count))
        self.safe_update_label(self.system_labels['uptime'], f"{elapsed:.1f}s")
        self.safe_update_label(self.system_labels['fps'], f"{fps:.1f} Hz")
        self.safe_update_label(self.system_labels['last_update'], time.strftime('%H:%M:%S'))
        
        self.root.after(100, self.process_queue)
    
    def start_monitoring(self):
        self.running = True
        self.monitor_thread_obj = threading.Thread(target=self.monitor_thread, daemon=True)
        self.monitor_thread_obj.start()
        self.process_queue()
    
    def on_closing(self):
        self.running = False
        if self.sock: self.sock.close()
        self.root.destroy()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = TorcsGUI()
    app.run()