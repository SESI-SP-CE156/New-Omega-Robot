"""
Módulo Principal de Visão e Controle - OBR
Este script atua como o "cérebro" de alto nível rodando no Raspberry Pi 5.
"""

import cv2
import numpy as np
import serial
import time
import csv
import os
from typing import Tuple, Optional
from enum import Enum, auto

# ==========================================
# CONFIGURAÇÕES FÍSICAS E DE CONTROLE
# ==========================================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

MAX_PWM = 250
MIN_MOTION_PWM = 110 
BASE_SPEED = 140

# ==========================================
# CONFIGURAÇÕES DA CÂMERA E GEOMETRIA
# ==========================================
CAM_WIDTH = 640   
CAM_HEIGHT = 480  
CAM_FPS = 10      
CAM_FOV_DEGREES = 90

BLIND_SPOT_CM = 6.0      
TOLERANCE_CM = 5.0       

VISION_WIDTH_CM_AT_BLIND_SPOT = 2 * BLIND_SPOT_CM * np.tan(np.radians(CAM_FOV_DEGREES / 2))
PIXELS_PER_CM = CAM_WIDTH / VISION_WIDTH_CM_AT_BLIND_SPOT
DEADZONE_PX = (TOLERANCE_CM / 2) * PIXELS_PER_CM

# Configurações de Cor para os Marcadores (HSV)
# Ajuste esses valores dependendo da luz da arena. Estes são padrões para verde.
LOWER_GREEN = np.array([40, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])
# Quantidade mínima de pixels verdes juntos para não confundir com ruído
MIN_GREEN_AREA = 800 

# Configurações de Cor para a Linha Vermelha (HSV wrap-around)
LOWER_RED_1 = np.array([0, 70, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 70, 50])
UPPER_RED_2 = np.array([180, 255, 255])
MIN_RED_AREA = 1500 # Área maior, pois a linha vermelha cruza a tela toda

# ==========================================
# GESTÃO DE SESSÃO E DADOS (DATASET)
# ==========================================
SESSION_DIR = f"data/sessao_{int(time.time())}"
os.makedirs(f"{SESSION_DIR}/images", exist_ok=True)

class RobotState(Enum):
    FOLLOWING_LINE = auto()
    HANDLING_GAP = auto()
    ALIGNING_TURN = auto()  # Anda reto um pouco para posicionar o eixo no cruzamento
    EXECUTING_TURN = auto() # Gira freneticamente no eixo até achar a linha de novo
    COURSE_FINISHED = auto() # Novo estado para a linha vermelha
    STOPPED = auto()

class PIDController:
    """Controlador Proporcional-Integral-Derivativo (PID)."""
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
            dt = 1e-4  

        p_out = self.kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100) 
        i_out = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative

        self.prev_error = error
        self.last_time = current_time

        return p_out + i_out + d_out

class StateManager:
    """
    Gerenciador da Máquina de Estados Finitos (FSM).
    Responsável por transitar entre rastreio, inércia (gap), curvas complexas (verdes) e fim (vermelho).
    """
    def __init__(self, gap_timeout_s: float = 1.2):
        self.state = RobotState.FOLLOWING_LINE
        self.gap_timer_start = 0.0
        self.gap_timeout_s = gap_timeout_s 
        
        self.turn_intent = "NONE"
        self.turn_timer_start = 0.0
        self.ALIGN_TIME_S = 1.25 
        self.BLIND_TURN_TIME_S = 0.95 
        
    # CORREÇÃO: Adicionamos red_detected na assinatura da função
    def update(self, line_detected: bool, green_status: str, cx_bottom: Optional[int], center: int, red_detected: bool) -> RobotState:
        current_time = time.time()
        
        match self.state:
            case RobotState.FOLLOWING_LINE:
                # CORREÇÃO: A lógica de checar o vermelho e verde fica AQUI dentro!
                if red_detected:
                    self.state = RobotState.COURSE_FINISHED
                    print(f"\n[{time.strftime('%H:%M:%S')}] [ESTADO] Linha Vermelha detectada! Fim do trajeto.")
                elif green_status != "NONE":
                    self.turn_intent = green_status
                    self.state = RobotState.ALIGNING_TURN
                    self.turn_timer_start = current_time
                    print(f"\n[{time.strftime('%H:%M:%S')}] [ESTADO] Marcador Verde ({green_status}) detectado! Alinhando para a curva...")
                elif not line_detected:
                    self.state = RobotState.HANDLING_GAP
                    self.gap_timer_start = current_time
                    print(f"\n[{time.strftime('%H:%M:%S')}] [ESTADO] Linha perdida! Iniciando inércia para atravessar o GAP.")
                    
            case RobotState.HANDLING_GAP:
                if line_detected:
                    self.state = RobotState.FOLLOWING_LINE
                    print(f"[{time.strftime('%H:%M:%S')}] [ESTADO] GAP superado! Retomando rastreio PID.\n")
                elif (current_time - self.gap_timer_start) > self.gap_timeout_s:
                    self.state = RobotState.STOPPED
                    print(f"[{time.strftime('%H:%M:%S')}] [ALERTA CRÍTICO] Tempo esgotado no GAP.")
            
            case RobotState.ALIGNING_TURN:
                if (current_time - self.turn_timer_start) > self.ALIGN_TIME_S:
                    self.state = RobotState.EXECUTING_TURN
                    self.turn_timer_start = current_time
                    print(f"[{time.strftime('%H:%M:%S')}] [ESTADO] Executando Pivot de curva...")
                    
            case RobotState.EXECUTING_TURN:
                time_in_turn = current_time - self.turn_timer_start
                if time_in_turn > self.BLIND_TURN_TIME_S:
                    if cx_bottom is not None:
                        dist_to_center = abs(cx_bottom - center)
                        if dist_to_center < (DEADZONE_PX * 2): 
                            self.state = RobotState.FOLLOWING_LINE
                            self.turn_intent = "NONE"
                            print(f"[{time.strftime('%H:%M:%S')}] [ESTADO] Curva concluída!\n")
                            
            case RobotState.STOPPED:
                if line_detected:
                    self.state = RobotState.FOLLOWING_LINE
                    print(f"\n[{time.strftime('%H:%M:%S')}] [ESTADO] Linha detectada novamente. Retomando missão.")
                    
        return self.state

class RobotController:
    """Gerencia a comunicação de hardware (ESP32) e a persistência de dados."""
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
        """Envia pacote P para aplicar PWM contínuo."""
        if self.ser:
            command = f"P,{right},{left}\n"
            self.ser.write(command.encode())
            
    def send_raw(self, command: str) -> None:
        """Envia pacotes complexos, como os de comando discreto M (Pivot) ou G."""
        if self.ser:
            self.ser.write(f"{command}\n".encode())

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
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS) 
    
    if hasattr(cv2, 'CAP_PROP_POWER_LINE_FREQUENCY'):
        cap.set(cv2.CAP_PROP_POWER_LINE_FREQUENCY, 2)
    if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    return cap

def process_vision(frame: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int], int, np.ndarray, str, np.ndarray, bool]:
    """
    Retorna: (cx_bottom, cx_mid, cx_top, center_x, thresh_black, green_status, mask_green)
    """
    # ====== DETECÇÃO DA LINHA PRETA ======
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    h_bottom_start, h_bottom_end = int(h * 0.70), h
    h_mid_start, h_mid_end = int(h * 0.40), int(h * 0.70)
    h_top_start, h_top_end = int(h * 0.10), int(h * 0.40)

    roi_bottom = thresh[h_bottom_start:h_bottom_end, :]
    roi_mid = thresh[h_mid_start:h_mid_end, :]
    roi_top = thresh[h_top_start:h_top_end, :]

    def get_centroid(roi_img):
        M = cv2.moments(roi_img)
        if M["m00"] > 0: return int(M["m10"] / M["m00"])
        return None

    cx_bottom = get_centroid(roi_bottom)
    cx_mid = get_centroid(roi_mid)
    cx_top = get_centroid(roi_top)

    # ====== DETECÇÃO DOS MARCADORES VERDES (OBR) ======
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # O verde deve ser buscado na metade inferior da tela para não capturar 
    # o cenário fora da arena ou camisas da arquibancada.
    mask_green_roi = mask_green[int(h*0.5):h, :]
    
    # Fatiamos verticalmente no meio para saber se o verde está na Esquerda ou Direita
    left_green = mask_green_roi[:, :w//2]
    right_green = mask_green_roi[:, w//2:]
    
    green_left_area = cv2.countNonZero(left_green)
    green_right_area = cv2.countNonZero(right_green)
    
    green_status = "NONE"

    # Lógica baseada em área. Se os dois lados possuem muitos pixels verdes, é Retorno de 180°
    if green_left_area > MIN_GREEN_AREA and green_right_area > MIN_GREEN_AREA:
        green_status = "BOTH"
    elif green_left_area > MIN_GREEN_AREA:
        green_status = "LEFT"
    elif green_right_area > MIN_GREEN_AREA:
        green_status = "RIGHT"

    # ====== DETECÇÃO DA LINHA VERMELHA (PARADA) ======
    mask_red_1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask_red_2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    # Focamos no centro inferior da tela para ter certeza de que a linha 
    # vermelha está no chão, bem debaixo do robô
    mask_red_roi = mask_red[int(h * 0.85):h, :]
    red_area = cv2.countNonZero(mask_red_roi)
    red_detected = (red_area > 800)

    return cx_bottom, cx_mid, cx_top, w // 2, thresh, green_status, mask_green, red_detected

def calculate_motor_speeds(correction: float, dynamic_base_speed: int) -> Tuple[int, int, str]:
    pwm_l = dynamic_base_speed + correction
    pwm_r = dynamic_base_speed - correction
    
    if pwm_l < MIN_MOTION_PWM: pwm_l = 0
    if pwm_r < MIN_MOTION_PWM: pwm_r = 0
        
    pwm_l = int(np.clip(pwm_l, 0, MAX_PWM))
    pwm_r = int(np.clip(pwm_r, 0, MAX_PWM))
    
    if abs(correction) < 20: 
        label = "F"
    else:
        label = "R" if correction > 0 else "L"
        
    return pwm_l, pwm_r, label

def main() -> None:
    cap = setup_camera()
    controller = RobotController()
    state_manager = StateManager(gap_timeout_s=1.2) 
    pid = PIDController(kp=5.0, ki=0.0, kd=0.0)

    print("==================================================")
    print(f"[INIT] Sistema de Visão OBR e FSM Iniciados.")
    print(f"[INIT] Área Útil (Deadzone): +/- {int(DEADZONE_PX)} pixels.")
    print("==================================================\n")

    HEADLESS_MODE = True 
    last_decision_log = None 
    target_frame_time = 1.0 / CAM_FPS 

    try:
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # CORREÇÃO: Apenas UMA chamada para process_vision, recebendo todas as variáveis
            cx_bottom, cx_mid, cx_top, center, thresh, green_status, mask_green, red_detected = process_vision(frame)
            
            # CORREÇÃO: Retornamos a variável line_detected que havia sido apagada
            line_detected = (cx_bottom is not None) or (cx_mid is not None)
            
            # Atualiza o estado
            current_state = state_manager.update(line_detected, green_status, cx_bottom, center, red_detected)

            match current_state:
                
                case RobotState.FOLLOWING_LINE:
                    # CORREÇÃO: Limpamos os ifs daqui. Se a FSM disse que é FOLLOWING_LINE, 
                    # apenas executamos o PID puro!
                    current_error = 0
                    dynamic_base_speed = BASE_SPEED

                    if cx_bottom is not None:
                        current_error = cx_bottom - center
                    elif cx_mid is not None:
                        current_error = cx_mid - center

                    if cx_top is not None and cx_bottom is not None:
                        curve_intensity = abs(cx_top - cx_bottom)
                        if curve_intensity > 40: 
                            speed_reduction = min(60, curve_intensity * 0.4) 
                            dynamic_base_speed = max(MIN_MOTION_PWM, int(BASE_SPEED - speed_reduction))

                    if abs(current_error) < DEADZONE_PX:
                        current_error = 0

                    correction = pid.compute(current_error)
                    pwm_l, pwm_r, label = calculate_motor_speeds(correction, dynamic_base_speed)
                    
                    controller.send_pwm(pwm_l, pwm_r)
                    controller.save_data(frame, pwm_l, pwm_r, label)

                    if label != last_decision_log:
                        last_decision_log = label
                        
                case RobotState.HANDLING_GAP:
                    pid.integral = 0 
                    safe_gap_speed = max(MIN_MOTION_PWM, BASE_SPEED - 20)
                    controller.send_pwm(safe_gap_speed, safe_gap_speed)
                    controller.save_data(frame, safe_gap_speed, safe_gap_speed, "G")
                    
                case RobotState.ALIGNING_TURN:
                    # Durante o alinhamento, ignora o erro e apenas segue em frente para cruzar a área verde
                    pid.integral = 0
                    align_speed = max(MIN_MOTION_PWM, BASE_SPEED - 10)
                    controller.send_pwm(align_speed, align_speed)
                    
                    # Rotula no dataset como 'T' (Turn Setup) para não poluir os dados de frente normal
                    controller.save_data(frame, align_speed, align_speed, "T")
                    
                case RobotState.EXECUTING_TURN:
                    # Não salvamos imagens durante o pivot para não poluir o dataset de AI 
                    # com imagens super borradas que ele não deve tentar imitar
                    pid.integral = 0
                    
                    # Manda o comando da firmware do Arduino (New-Omega.ino)
                    if state_manager.turn_intent == "LEFT":
                        controller.send_raw("M,L")
                    elif state_manager.turn_intent == "RIGHT":
                        controller.send_raw("M,R")
                    elif state_manager.turn_intent == "BOTH":
                        # Giro 180 usa um lado fixo até achar a linha traseira
                        controller.send_raw("M,REV")

                case RobotState.STOPPED:
                    pid.integral = 0
                    controller.send_pwm(0, 0)
                    last_decision_log = "LOST"
                
                case RobotState.COURSE_FINISHED:
                    # Trava definitiva. O robô não tenta mais voltar para FOLLOWING_LINE
                    pid.integral = 0
                    controller.send_pwm(0, 0)
                    
                    # Anotamos isso no dataset como 'END'
                    controller.save_data(frame, 0, 0, "END")

            # ==========================================
            # DEBUG E VISUALIZAÇÃO
            # ==========================================
            if not HEADLESS_MODE:
                cv2.line(frame, (int(center - DEADZONE_PX), 0), (int(center - DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                cv2.line(frame, (int(center + DEADZONE_PX), 0), (int(center + DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                
                if cx_bottom is not None: cv2.circle(frame, (cx_bottom, int(CAM_HEIGHT * 0.85)), 8, (0, 255, 0), -1)
                if cx_mid is not None: cv2.circle(frame, (cx_mid, int(CAM_HEIGHT * 0.55)), 8, (0, 255, 255), -1)
                if cx_top is not None: cv2.circle(frame, (cx_top, int(CAM_HEIGHT * 0.25)), 8, (0, 0, 255), -1)

                cv2.imshow("Sistema de Visao - Binarizado", thresh)
                cv2.imshow("Sistema de Visao - Mascara Verde", mask_green)
                cv2.imshow("Sistema de Visao - RGB", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

            # Controle de Cadência de Frame (FPS)
            processing_time = time.time() - loop_start_time
            if processing_time < target_frame_time:
                time.sleep(target_frame_time - processing_time)

    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()