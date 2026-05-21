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
MAX_PWM = 140
MIN_PWM = 140
BASE_SPEED = 140
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

class PIDController:
    """
    Encapsula a lógica do controle Proporcional, Integral e Derivativo.
    Mantém o estado do erro anterior e o acúmulo da integral de forma limpa.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error: float) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0.0:
            dt = 1e-4  # Previne divisão por zero em loops ultra-rápidos

        # Proporcional
        p_out = self.kp * error

        # Integral (com Anti-windup para limitar o acúmulo infinito)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100) 
        i_out = self.ki * self.integral

        # Derivativo (Taxa de variação do erro)
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative

        # Atualiza o estado para o próximo loop
        self.prev_error = error
        self.last_time = current_time

        return p_out + i_out + d_out

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

def process_vision(frame: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int], int, np.ndarray]:
    """
    Fatia a imagem em 3 ROIs independentes para permitir visão preditiva.
    Retorna os centroides: (cx_bottom, cx_mid, cx_top, center_x, imagem_binarizada)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    
    # Define as alturas de cada ROI
    h_bottom_start, h_bottom_end = int(h * 0.70), h
    h_mid_start, h_mid_end = int(h * 0.40), int(h * 0.70)
    h_top_start, h_top_end = int(h * 0.10), int(h * 0.40)

    roi_bottom = thresh[h_bottom_start:h_bottom_end, :]
    roi_mid = thresh[h_mid_start:h_mid_end, :]
    roi_top = thresh[h_top_start:h_top_end, :]

    def get_centroid(roi_img):
        M = cv2.moments(roi_img)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])
        return None

    cx_bottom = get_centroid(roi_bottom)
    cx_mid = get_centroid(roi_mid)
    cx_top = get_centroid(roi_top)

    return cx_bottom, cx_mid, cx_top, w // 2, thresh

def calculate_motor_speeds(correction: float, dynamic_base_speed: int) -> Tuple[int, int, str]:
    """
    Calcula as velocidades dos motores recebendo a saída do PID e a velocidade dinâmica.
    """
    # A correção é submetida ao int() e somada/subtraída da velocidade base adaptativa
    pwm_l = int(np.clip(dynamic_base_speed + correction, MIN_PWM, MAX_PWM))
    pwm_r = int(np.clip(dynamic_base_speed - correction, MIN_PWM, MAX_PWM))
    
    # Threshold um pouco maior para evitar marcar retas como curvas leves no dataset
    if abs(correction) < 15:
        label = "F"
    else:
        label = "R" if correction > 0 else "L"
        
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

    HEADLESS_MODE = True 
    last_decision_log = None 

    # Intervalo alvo para manter exatos 10 FPS (1.0 segundo / 10 = 0.1s por frame)
    target_frame_time = 1.0 / CAM_FPS

    # Antes do try, instancie o controlador PID:
    # Ajuste esses valores na arena! Sugestão inicial: P forte, D médio, I baixo.
    pid = PIDController(kp=0.8, ki=0.01, kd=0.15)

    try:
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # Captura os múltiplos ROIs
            cx_bottom, cx_mid, cx_top, center, thresh = process_vision(frame)

            current_error = 0
            dynamic_base_speed = BASE_SPEED

            # 1. Prioridade de Rastreio: Tenta seguir a parte de baixo, se perder, tenta o meio.
            if cx_bottom is not None:
                current_error = cx_bottom - center
            elif cx_mid is not None:
                current_error = cx_mid - center

            # 2. Visão Preditiva: Freio Inteligente em Curvas
            if cx_top is not None and cx_bottom is not None:
                # Se a diferença entre a linha lá na frente e a linha de baixo for muito grande, é uma curva
                curve_intensity = abs(cx_top - cx_bottom)
                if curve_intensity > 40: # Threshold de curva (Ajuste na pista)
                    # Reduz a velocidade com base na agressividade da curva
                    speed_reduction = min(60, curve_intensity * 0.4) 
                    dynamic_base_speed = max(90, int(BASE_SPEED - speed_reduction))

            # 3. Aplica a zona morta (Deadzone)
            if abs(current_error) < DEADZONE_PX:
                current_error = 0

            # 4. Cálculo final (PID + Conversão para Motores)
            if cx_bottom is not None or cx_mid is not None:
                correction = pid.compute(current_error)
                pwm_l, pwm_r, label = calculate_motor_speeds(correction, dynamic_base_speed)
                
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
                # Linha totalmente perdida (Nenhum ROI detectou nada)
                pid.integral = 0 # Reseta a memória do PID por segurança
                controller.send_pwm(0, 0)
                
                if last_decision_log != "LOST":
                    print(f"[{time.strftime('%H:%M:%S')}] [ALERTA CRÍTICO] Linha perdida! Motores parados por segurança.")
                    last_decision_log = "LOST"

            # ==========================================
            # DEBUG E VISUALIZAÇÃO
            # ==========================================
            if not HEADLESS_MODE:
                # Desenha o centro desejado e a zona morta
                cv2.line(frame, (int(center - DEADZONE_PX), 0), (int(center - DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                cv2.line(frame, (int(center + DEADZONE_PX), 0), (int(center + DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                
                # Desenha os centroides rastreados (Bolinha Verde = Bottom, Amarela = Mid, Vermelha = Top)
                if cx_bottom is not None: cv2.circle(frame, (cx_bottom, int(CAM_HEIGHT * 0.85)), 8, (0, 255, 0), -1)
                if cx_mid is not None: cv2.circle(frame, (cx_mid, int(CAM_HEIGHT * 0.55)), 8, (0, 255, 255), -1)
                if cx_top is not None: cv2.circle(frame, (cx_top, int(CAM_HEIGHT * 0.25)), 8, (0, 0, 255), -1)

                # Exibe a imagem binarizada para depuração de iluminação junto com o RGB
                cv2.imshow("Sistema de Visao - Binarizado", thresh)
                cv2.imshow("Sistema de Visao - RGB", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): 
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