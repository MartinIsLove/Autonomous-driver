import pygame
import time
import sys

def test_joystick():
    # Inizializza pygame
    pygame.init()
    pygame.joystick.init()
    
    # Controlla se ci sono joystick connessi
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("Nessun controller trovato!")
        return
    
    print(f"Trovati {joystick_count} controller(s)")
    
    # Mostra informazioni su tutti i controller
    for i in range(joystick_count):
        js = pygame.joystick.Joystick(i)
        print(f"Controller {i}: {js.get_name()}")
    
    # Usa il primo joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"\nUsando controller: {joystick.get_name()}")
    print(f"Numero di assi: {joystick.get_numaxes()}")
    print(f"Numero di bottoni: {joystick.get_numbuttons()}")
    print(f"Numero di hat: {joystick.get_numhats()}")
    print(f"Controller inizializzato: {joystick.get_init()}")
    
    # Inizializza valori precedenti degli assi
    prev_axes = [0.0] * joystick.get_numaxes()
    threshold = 0.01  # Soglia per considerare un movimento significativo
    
    # Test di connessione
    print("\nTest di connessione controller...")
    for i in range(5):
        pygame.event.pump()
        print(f"Test {i+1}: Asse 0 = {joystick.get_axis(0):.3f}")
        time.sleep(0.5)
    
    print("\nPremi Ctrl+C per uscire")
    print("Muovi il controller per vedere i valori...\n")
    
    try:
        while True:
            # Processa tutti gli eventi pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"Bottone {event.button} premuto!")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Bottone {event.button} rilasciato!")
                elif event.type == pygame.JOYAXISMOTION:
                    print(f"Asse {event.axis} mosso: {event.value:.3f}")
            
            pygame.event.pump()
            
            # Controlla quali assi sono cambiati
            moving_axes = []
            for i in range(joystick.get_numaxes()):
                current_value = joystick.get_axis(i)
                if abs(current_value - prev_axes[i]) > threshold:
                    moving_axes.append((i, current_value))
                prev_axes[i] = current_value
            
            # Mostra solo gli assi che si muovono
            if moving_axes:
                print("\033[H\033[J", end="")  # Pulisci lo schermo
                print(f"Controller: {joystick.get_name()}")
                print("=" * 50)
                print("ASSI IN MOVIMENTO:")
                for axis_num, value in moving_axes:
                    bar = "█" * int(abs(value) * 20)
                    print(f"  Asse {axis_num}: {value:6.3f} |{bar:<20}|")

            
            time.sleep(0.1)  # Aggiorna 10 volte al secondo
            
    except KeyboardInterrupt:
        print("\nTest interrotto dall'utente")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    test_joystick()