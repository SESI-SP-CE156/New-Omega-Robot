import cv2
import numpy as np
import serial
import time
import csv
import os
from collections import deque
from typing import Tuple, Optional

# ==========================================
# CONFIGURAÇÕES FÍSICAS E DE CONTROLE
# ==========================================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
MAX_PWM = 150
MIN_PWM = 80
BASE_SPEED = 120
KP = 0.8  # Ganho Proporcional

# ==========================================
# CONFIGURAÇÕES DA CÂMERA E GEOMETRIA
# ==========================================
CAM_WIDTH = 640   
CAM_HEIGHT = 480  
CAM_FPS = 10      # Reduzido de 30 para 10 FPS
CAM_FOV_DEGREES = 90

BLIND_SPOT_CM = 6.0      
TOLERANCE_CM = 5.0       
ROBOT_SPEED_CM_S = 25.0  

VISION_WIDTH_CM_AT_BLIND_SPOT = 2 * BLIND_SPOT_CM * np.tan(np.radians(CAM_FOV_DEGREES / 2))
PIXELS_PER_CM = CAM_WIDTH / VISION_WIDTH_CM_AT_BLIND_SPOT

DEADZONE_PX = (TOLERANCE_CM / 2) * PIXELS_PER_CM

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
            time.sleep(2) 
            print("[SISTEMA] Controlador conectado com sucesso.")
        except Exception as e:
            print(f"[AVISO] Erro ao conectar no microcontrolador: {e}")
            self.ser = None

        self.log_file = open(f"{SESSION_DIR}/labels.csv", mode='w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["img_path", "pwm_l", "pwm_r", "label"])

    def send_pwm(self, left: int, right: int) -> None:
        if self.ser:
            command = f"P,{right},{left}\n"
            self.ser.write(command.encode())

    def save_data(self, frame: np.ndarray, l_pwm: int, r_pwm: int, label: str) -> None:
        timestamp = int(time.time() * 1000)
        img_name = f"img_{timestamp}.jpg"
        img_path = f"images/{img_name}"
        
        cv2.imwrite(f"{SESSION_DIR}/{img_path}", frame)
        self.writer.writerow([img_path, r_pwm, l_pwm, label])

    def close(self) -> None:
        self.log_file.close()
        if self.ser:
            self.send_pwm(0, 0)
            self.ser.close()
        print("[SISTEMA] Conexão encerrada e arquivos salvos.")

def setup_camera() -> cv2.VideoCapture:
    """Configura a câmera forçando o backend nativo do Linux (V4L2)."""
    # Adicionada a flag cv2.CAP_V4L2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    # Solicita a mudança para o hardware (nem todas as câmeras respeitam)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS) 
    
    if hasattr(cv2, 'CAP_PROP_POWER_LINE_FREQUENCY'):
        cap.set(cv2.CAP_PROP_POWER_LINE_FREQUENCY, 2)
    
    if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    return cap

def process_vision(frame: np.ndarray) -> Tuple[Optional[int], int, Optional[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    roi_start = int(h * 0.8) 
    roi = thresh[roi_start:h, :]

    M = cv2.moments(roi)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        return cx, w // 2, roi
    return None, w // 2, None

def calculate_motor_speeds(active_error: float) -> Tuple[int, int, str]:
    if active_error == 0:
        return BASE_SPEED, BASE_SPEED, "F"
    
    correction = int(active_error * KP)
    pwm_l = int(np.clip(BASE_SPEED + correction, MIN_PWM, MAX_PWM))
    pwm_r = int(np.clip(BASE_SPEED - correction, MIN_PWM, MAX_PWM))
    
    label = "R" if active_error > 0 else "L"
    return pwm_l, pwm_r, label

def main() -> None:
    cap = setup_camera()
    controller = RobotController()
    
    error_queue = deque([0] * DELAY_FRAMES, maxlen=DELAY_FRAMES)

    print("==================================================")
    print(f"[INIT] Sistema de Visão Iniciado.")
    print(f"[INIT] Câmera configurada para: {CAM_FPS} FPS.")
    print(f"[INIT] Delay compensatório: {DELAY_FRAMES} frames.")
    print(f"[INIT] Área Útil (Deadzone): +/- {int(DEADZONE_PX)} pixels.")
    print("==================================================\n")

    HEADLESS_MODE = False 
    last_decision_log = None 

    # Intervalo alvo para manter exatos 10 FPS (1.0 segundo / 10 = 0.1s por frame)
    target_frame_time = 1.0 / CAM_FPS

    try:
        while True:
            # Inicia o cronômetro do frame
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret: 
                break

            cx, center, _ = process_vision(frame)

            current_visual_error = 0
            if cx is not None:
                raw_error = cx - center
                if abs(raw_error) > DEADZONE_PX:
                    current_visual_error = raw_error

            error_queue.append(current_visual_error)
            active_error = error_queue[0]

            if cx is not None:
                pwm_l, pwm_r, label = calculate_motor_speeds(active_error)
                controller.send_pwm(pwm_l, pwm_r)
                controller.save_data(frame, pwm_l, pwm_r, label)
                
                if label != last_decision_log:
                    if label == "F":
                        print(f"[{time.strftime('%H:%M:%S')}] [AÇÃO] Linha na área útil. Andando para FRENTE (PWM: {pwm_l}/{pwm_r}).")
                    elif label == "R":
                        print(f"[{time.strftime('%H:%M:%S')}] [AÇÃO] Corrigindo posição para a DIREITA (PWM Esq: {pwm_l}, Dir: {pwm_r}).")
                    elif label == "L":
                        print(f"[{time.strftime('%H:%M:%S')}] [AÇÃO] Corrigindo posição para a ESQUERDA (PWM Esq: {pwm_l}, Dir: {pwm_r}).")
                    last_decision_log = label

            else:
                controller.send_pwm(0, 0)
                
                if last_decision_log != "LOST":
                    print(f"[{time.strftime('%H:%M:%S')}] [ALERTA CRÍTICO] Linha perdida! Motores parados por segurança.")
                    last_decision_log = "LOST"

            # ==========================================
            # DEBUG E VISUALIZAÇÃO
            # ==========================================
            if not HEADLESS_MODE:
                cv2.line(frame, (int(center - DEADZONE_PX), 0), (int(center - DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 2)
                cv2.line(frame, (int(center + DEADZONE_PX), 0), (int(center + DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 2)
                
                if cx is not None: 
                    color = (0, 255, 0) if current_visual_error == 0 else (0, 0, 255)
                    cv2.circle(frame, (cx, int(CAM_HEIGHT * 0.9)), 10, color, -1)
                
                cv2.imshow("Sistema de Visao - OBR", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    print("\n[SISTEMA] Desligamento solicitado pelo usuário.")
                    break

            # ==========================================
            # CONTROLE DE CADÊNCIA (FPS THROTTLE)
            # ==========================================
            processing_time = time.time() - loop_start_time
            # Se o processamento for mais rápido que o tempo limite do frame, o sistema "dorme" a diferença
            if processing_time < target_frame_time:
                time.sleep(target_frame_time - processing_time)

    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()