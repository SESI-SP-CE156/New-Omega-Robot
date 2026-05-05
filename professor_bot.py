import cv2
import numpy as np
import serial
import time
import csv
import os
from collections import deque

# ==========================================
# CONFIGURAÇÕES FÍSICAS E DE CONTROLE
# ==========================================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
MAX_PWM = 150
BASE_SPEED = 120
KP = 0.8  # Ganho Proporcional

# ==========================================
# CONFIGURAÇÕES DA CÂMERA E GEOMETRIA
# ==========================================
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
CAM_FOV_DEGREES = 90

BLIND_SPOT_CM = 6.0      # Distância cega à frente do robô
TOLERANCE_CM = 5.0       # Área útil desejada no centro (linha tem 2cm, folga de 5cm)
ROBOT_SPEED_CM_S = 25.0  # VELOCIDADE ESTIMADA DO ROBÔ (cm por segundo) - Ajuste isso!

# Cálculo de Proporção Pixel -> Centímetro
# Considerando FOV de 90º, a largura (W) visualizada a uma distância (D) é 2 * D * tan(45º)
# A 6cm de distância, a largura do quadro é aprox 12cm.
VISION_WIDTH_CM_AT_BLIND_SPOT = 2 * BLIND_SPOT_CM * np.tan(np.radians(CAM_FOV_DEGREES / 2))
PIXELS_PER_CM = CAM_WIDTH / VISION_WIDTH_CM_AT_BLIND_SPOT

# Limite em pixels para a área útil (metade para cada lado do centro)
DEADZONE_PX = (TOLERANCE_CM / 2) * PIXELS_PER_CM

# Cálculo de Frames para o Atraso (Delay da Zona Cega)
# Tempo para percorrer a zona cega = Distância / Velocidade
TIME_TO_BLIND_SPOT_S = BLIND_SPOT_CM / ROBOT_SPEED_CM_S
DELAY_FRAMES = max(1, int(TIME_TO_BLIND_SPOT_S * CAM_FPS))

# ==========================================
# GESTÃO DE SESSÃO E DADOS
# ==========================================
SESSION_DIR = f"data/sessao_{int(time.time())}"
os.makedirs(f"{SESSION_DIR}/images", exist_ok=True)

class RobotController:
    def __init__(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2) # Aguarda boot do ESP32
            print("Controlador conectado.")
        except Exception as e:
            print(f"Aviso: Erro ao conectar no microcontrolador: {e}")
            self.ser = None

        self.log_file = open(f"{SESSION_DIR}/labels.csv", mode='w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["img_path", "pwm_l", "pwm_r", "label"])

    def send_pwm(self, left: int, right: int):
        """Envia comandos de velocidade de forma linear, garantindo simetria nos motores."""
        if self.ser:
            command = f"P,{left},{right}\n"
            self.ser.write(command.encode())

    def save_data(self, frame: np.ndarray, l_pwm: int, r_pwm: int, label: str):
        timestamp = int(time.time() * 1000)
        img_name = f"img_{timestamp}.jpg"
        img_path = f"images/{img_name}"
        
        cv2.imwrite(f"{SESSION_DIR}/{img_path}", frame)
        self.writer.writerow([img_path, l_pwm, r_pwm, label])

    def close(self):
        self.log_file.close()
        if self.ser:
            self.send_pwm(0, 0)
            self.ser.close()

def setup_camera() -> cv2.VideoCapture:
    """Configura a câmera com as especificações exigidas de hardware."""
    cap = cv2.VideoCapture(0)
    
    # Forçar Resolução e FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    
    # Tratamento defensivo: Aplica correção de 60Hz apenas se a constante existir na versão atual do OpenCV
    if hasattr(cv2, 'CAP_PROP_POWER_LINE_FREQUENCY'):
        cap.set(cv2.CAP_PROP_POWER_LINE_FREQUENCY, 2)
    else:
        print("[Aviso] Constante de Power Line Frequency não encontrada no OpenCV. Ignorando...")
    
    # Tratamento defensivo para o foco automático
    if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    return cap

def process_vision(frame: np.ndarray):
    """Processa a imagem e retorna o centróide e a máscara."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    # Pegamos uma faixa inferior da imagem (ex: 20% inferiores)
    h, w = thresh.shape
    roi_start = int(h * 0.8) 
    roi = thresh[roi_start:h, :]

    M = cv2.moments(roi)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        return cx, w // 2, roi
    return None, w // 2, None

def main():
    cap = setup_camera()
    controller = RobotController()
    
    # Fila para gerenciar o atraso da zona cega
    # Inicializada com zeros (sem erro no começo)
    error_queue = deque([0] * DELAY_FRAMES, maxlen=DELAY_FRAMES)

    print(f"Sistema Iniciado. Delay compensatório: {DELAY_FRAMES} frames.")
    print(f"Deadzone (Área Útil): +/- {int(DEADZONE_PX)} pixels do centro.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                break

            cx, center, _ = process_vision(frame)

            # 1. Obtenção do Erro Visual (O que a câmera vê AGORA a 6cm de distância)
            current_visual_error = 0
            if cx is not None:
                raw_error = cx - center
                
                # Aplicação da Área Útil (Deadzone de 5cm)
                if abs(raw_error) > DEADZONE_PX:
                    current_visual_error = raw_error

            # 2. Alimentar a fila de memória do robô
            error_queue.append(current_visual_error)

            # 3. Execução do Controle (Lendo o erro que a câmera viu `DELAY_FRAMES` atrás)
            # Dessa forma, o robô só age quando a roda estiver de fato passando pela área lida.
            active_error = error_queue[0]

            if cx is not None:
                correction = int(active_error * KP)

                # Ajuste de PWM Dinâmico Proporcional
                pwm_l = np.clip(BASE_SPEED + correction, 0, MAX_PWM)
                pwm_r = np.clip(BASE_SPEED - correction, 0, MAX_PWM)

                controller.send_pwm(pwm_l, pwm_r)
                
                # Classificação para a IA (Opcional baseada no erro ativo)
                label = "F" if active_error == 0 else ("R" if active_error > 0 else "L")
                controller.save_data(frame, pwm_l, pwm_r, label)
            else:
                controller.send_pwm(0, 0) # Segurança caso perca a linha completamente

            # ==========================================
            # DEBUG E VISUALIZAÇÃO
            # ==========================================
            # Dica de ouro: Na arena, defina HEADLESS_MODE = True
            HEADLESS_MODE = False 

            if not HEADLESS_MODE:
                # Desenha limites da área útil (Deadzone)
                cv2.line(frame, (int(center - DEADZONE_PX), 0), (int(center - DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 2)
                cv2.line(frame, (int(center + DEADZONE_PX), 0), (int(center + DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 2)
                
                # Desenha o centróide visual atual
                if cx: 
                    color = (0, 255, 0) if current_visual_error == 0 else (0, 0, 255)
                    cv2.circle(frame, (cx, int(CAM_HEIGHT * 0.9)), 10, color, -1)
                
                cv2.imshow("Sistema de Visao - OBR", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()