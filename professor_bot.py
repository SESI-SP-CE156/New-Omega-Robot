import cv2
import numpy as np
import serial
import time
import csv
import os
from typing import Tuple, Optional

# ==========================================
# CONFIGURAÇÕES FÍSICAS E DE CONTROLE
# ==========================================
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
MAX_PWM = 250

# NOVA CONFIGURAÇÃO: O limite físico onde o motor consegue empurrar o robô
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

# ==========================================
# GESTÃO DE SESSÃO E DADOS
# ==========================================
SESSION_DIR = f"data/sessao_{int(time.time())}"
os.makedirs(f"{SESSION_DIR}/images", exist_ok=True)

class PIDController:
    """
    Controlador Proporcional-Integral-Derivativo (PID).
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

class RobotController:
    """
    Gerencia a comunicação de hardware (Serial com o ESP32) e a persistência de dados.
    """
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
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS) 
    
    if hasattr(cv2, 'CAP_PROP_POWER_LINE_FREQUENCY'):
        cap.set(cv2.CAP_PROP_POWER_LINE_FREQUENCY, 2)
    if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    return cap

def process_vision(frame: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int], int, np.ndarray]:
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
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])
        return None

    cx_bottom = get_centroid(roi_bottom)
    cx_mid = get_centroid(roi_mid)
    cx_top = get_centroid(roi_top)

    return cx_bottom, cx_mid, cx_top, w // 2, thresh

def calculate_motor_speeds(correction: float, dynamic_base_speed: int) -> Tuple[int, int, str]:
    """
    Aplica a correção do PID, respeitando o limite máximo e a Zona Morta dos motores.
    """
    pwm_l = dynamic_base_speed + correction
    pwm_r = dynamic_base_speed - correction
    
    # --- LOGICA DE ZONA MORTA ---
    # Se o valor calculado for menor que o necessário para o motor ter força (110),
    # cortamos imediatamente para 0. Isso transforma "curvas arrastadas" em "curvas fechadas/pivot".
    if pwm_l < MIN_MOTION_PWM:
        pwm_l = 0
    if pwm_r < MIN_MOTION_PWM:
        pwm_r = 0
        
    # Garante que nenhum valor passe do teto máximo de segurança (250)
    pwm_l = int(np.clip(pwm_l, 0, MAX_PWM))
    pwm_r = int(np.clip(pwm_r, 0, MAX_PWM))
    
    # Rotulação para a IA (threshold aumentado para lidar com oscilações normais em retas)
    if abs(correction) < 20: 
        label = "F"
    else:
        label = "R" if correction > 0 else "L"
        
    return pwm_l, pwm_r, label

def main() -> None:
    cap = setup_camera()
    controller = RobotController()

    print("==================================================")
    print(f"[INIT] Sistema de Visão Iniciado.")
    print(f"[INIT] Câmera configurada para: {CAM_FPS} FPS.")
    print(f"[INIT] Área Útil (Deadzone): +/- {int(DEADZONE_PX)} pixels.")
    print("==================================================\n")

    HEADLESS_MODE = True 
    last_decision_log = None 
    target_frame_time = 1.0 / CAM_FPS

    pid = PIDController(kp=5.0, ki=0.0, kd=0.0)

    try:
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            cx_bottom, cx_mid, cx_top, center, thresh = process_vision(frame)

            current_error = 0
            dynamic_base_speed = BASE_SPEED

            if cx_bottom is not None:
                current_error = cx_bottom - center
            elif cx_mid is not None:
                current_error = cx_mid - center

            # 2. Visão Preditiva (Freio Inteligente): 
            if cx_top is not None and cx_bottom is not None:
                curve_intensity = abs(cx_top - cx_bottom)
                if curve_intensity > 40: 
                    speed_reduction = min(60, curve_intensity * 0.4) 
                    
                    # SEGURANÇA DE FREIO: A base NUNCA cai abaixo de MIN_MOTION_PWM (110)
                    # Caso contrário, o robô vai frear tanto para a curva que não terá força para entrar nela.
                    dynamic_base_speed = max(MIN_MOTION_PWM, int(BASE_SPEED - speed_reduction))

            if abs(current_error) < DEADZONE_PX:
                current_error = 0

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
                pid.integral = 0 
                controller.send_pwm(0, 0)
                
                if last_decision_log != "LOST":
                    print(f"[{time.strftime('%H:%M:%S')}] [ALERTA CRÍTICO] Linha perdida! Motores parados por segurança.")
                    last_decision_log = "LOST"

            if not HEADLESS_MODE:
                cv2.line(frame, (int(center - DEADZONE_PX), 0), (int(center - DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                cv2.line(frame, (int(center + DEADZONE_PX), 0), (int(center + DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                
                if cx_bottom is not None: cv2.circle(frame, (cx_bottom, int(CAM_HEIGHT * 0.85)), 8, (0, 255, 0), -1)
                if cx_mid is not None: cv2.circle(frame, (cx_mid, int(CAM_HEIGHT * 0.55)), 8, (0, 255, 255), -1)
                if cx_top is not None: cv2.circle(frame, (cx_top, int(CAM_HEIGHT * 0.25)), 8, (0, 0, 255), -1)

                cv2.imshow("Sistema de Visao - Binarizado", thresh)
                cv2.imshow("Sistema de Visao - RGB", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

            processing_time = time.time() - loop_start_time
            if processing_time < target_frame_time:
                time.sleep(target_frame_time - processing_time)

    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()