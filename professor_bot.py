"""
Módulo Principal de Visão e Controle - OBR
Este script atua como o "cérebro" de alto nível rodando no Raspberry Pi 5.
Responsabilidades:
1. Capturar e processar imagens da câmera (visão computacional).
2. Calcular correções de trajetória usando controle PID e visão preditiva (3 ROIs).
3. Gerenciar o problema mecânico de Zona Morta (Estol) dos motores.
4. Enviar comandos de velocidade (PWM) via Serial para o ESP32 (atuador de baixo nível).
5. Gravar os frames e decisões em um dataset (CSV) para o futuro treinamento de IA (AI HAT+).
"""

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

# Limite máximo de segurança para a Ponte-H e os motores
MAX_PWM = 250

# ZONA MORTA MECÂNICA (Torque de Estol): 
# Valor mínimo de PWM necessário para vencer o atrito estático do motor e mover o peso do robô.
# Valores abaixo deste apenas aquecerão o motor sem gerar movimento.
MIN_MOTION_PWM = 110 
BASE_SPEED = 140

# ==========================================
# CONFIGURAÇÕES DA CÂMERA E GEOMETRIA
# ==========================================
CAM_WIDTH = 640   
CAM_HEIGHT = 480  
CAM_FPS = 10      
CAM_FOV_DEGREES = 90

# Distância do ponto cego (chassi tapando a visão) e tolerância de erro
BLIND_SPOT_CM = 6.0      
TOLERANCE_CM = 5.0       

# Cinemática Inversa simples: Calcula quantos pixels na tela representam 1 cm no mundo real
VISION_WIDTH_CM_AT_BLIND_SPOT = 2 * BLIND_SPOT_CM * np.tan(np.radians(CAM_FOV_DEGREES / 2))
PIXELS_PER_CM = CAM_WIDTH / VISION_WIDTH_CM_AT_BLIND_SPOT

# Define a "Zona Morta" visual em pixels (área onde o robô é considerado centralizado)
DEADZONE_PX = (TOLERANCE_CM / 2) * PIXELS_PER_CM

# ==========================================
# GESTÃO DE SESSÃO E DADOS (DATASET)
# ==========================================
# Cria um diretório isolado por timestamp para evitar sobrescrever dados de treinos anteriores
SESSION_DIR = f"data/sessao_{int(time.time())}"
os.makedirs(f"{SESSION_DIR}/images", exist_ok=True)

class PIDController:
    """
    Controlador Proporcional-Integral-Derivativo (PID).
    Calcula a força necessária para corrigir o desvio do robô em relação à linha.
    - P: Reage ao erro atual (Força principal).
    - I: Corrige erros crônicos (ex: um motor mais fraco que o outro).
    - D: Amortece o movimento, evitando que o robô balance feito um pêndulo.
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
        
        # Trava de segurança para evitar divisão por zero
        if dt <= 0.0:
            dt = 1e-4  

        # [Proporcional]
        p_out = self.kp * error

        # [Integral] Acúmulo no tempo. Usa 'clip' como Anti-Windup para evitar crescimento infinito
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100) 
        i_out = self.ki * self.integral

        # [Derivativo] Taxa de variação (velocidade com que o erro muda)
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative

        # Salva o estado para o próximo cálculo
        self.prev_error = error
        self.last_time = current_time

        return p_out + i_out + d_out

class RobotController:
    """
    Gerencia a camada de I/O do sistema: 
    1. Envio de comandos seriais para o firmware do ESP32.
    2. Escrita do log (CSV) e salvamento de imagens para Deep Learning.
    """
    def __init__(self):
        # Tenta conectar com o microcontrolador ESP32
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2) # Aguarda o boot/reset automático do ESP ao abrir a serial
            print("[SISTEMA] Controlador conectado com sucesso.")
        except Exception as e:
            print(f"[AVISO] Erro ao conectar no microcontrolador: {e}")
            self.ser = None

        # Inicializa o arquivo de anotações (labels) que alimentará a futura rede neural
        self.log_file = open(f"{SESSION_DIR}/labels.csv", mode='w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["img_path", "pwm_l", "pwm_r", "label"])

    def send_pwm(self, left: int, right: int) -> None:
        """Monta o pacote de dados via protocolo de texto e despacha para o ESP32."""
        if self.ser:
            command = f"P,{right},{left}\n"
            self.ser.write(command.encode())

    def save_data(self, frame: np.ndarray, l_pwm: int, r_pwm: int, label: str) -> None:
        """Persiste o frame capturado no disco e atrela a decisão correspondente no CSV."""
        timestamp = int(time.time() * 1000)
        img_name = f"img_{timestamp}.jpg"
        img_path = f"images/{img_name}"
        
        cv2.imwrite(f"{SESSION_DIR}/{img_path}", frame)
        self.writer.writerow([img_path, r_pwm, l_pwm, label])

    def close(self) -> None:
        """Garante a parada física do robô e o fechamento íntegro dos arquivos ao sair."""
        self.log_file.close()
        if self.ser:
            self.send_pwm(0, 0)
            self.ser.close()
        print("[SISTEMA] Conexão encerrada e arquivos salvos.")

def setup_camera() -> cv2.VideoCapture:
    """
    Inicia a câmera exigindo a API V4L2 nativa do Linux. 
    Isso é crucial para que parâmetros de hardware (Anti-flicker e Exposição Manual) 
    configurados no script 'setup_camera.sh' não sejam sobrescritos pelo OpenCV.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS) 
    
    # Desativa recursos automáticos que causam oscilação na leitura da linha
    if hasattr(cv2, 'CAP_PROP_POWER_LINE_FREQUENCY'):
        cap.set(cv2.CAP_PROP_POWER_LINE_FREQUENCY, 2)
    if hasattr(cv2, 'CAP_PROP_AUTOFOCUS'):
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    return cap

def process_vision(frame: np.ndarray) -> Tuple[Optional[int], Optional[int], Optional[int], int, np.ndarray]:
    """
    Pipeline de Visão Computacional Clássica.
    Fatia a imagem em 3 ROIs (Region of Interest) - Bottom, Mid e Top.
    Isso simula 'profundidade' e permite que o robô saiba de curvas antecipadamente.
    """
    # 1. Filtros morfológicos para isolar a linha preta do chão claro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    
    # 2. Definição das coordenadas verticais (Y) das fatias
    h_bottom_start, h_bottom_end = int(h * 0.70), h
    h_mid_start, h_mid_end = int(h * 0.40), int(h * 0.70)
    h_top_start, h_top_end = int(h * 0.10), int(h * 0.40)

    # 3. Recorte do mapa binário
    roi_bottom = thresh[h_bottom_start:h_bottom_end, :]
    roi_mid = thresh[h_mid_start:h_mid_end, :]
    roi_top = thresh[h_top_start:h_top_end, :]

    def get_centroid(roi_img):
        """Calcula o Momento Espacial para encontrar o Centro de Massa da linha detectada."""
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
    Converte o erro numérico do PID em pulsos PWM para as rodas.
    Aplica ativamente as regras de limite mecânico (Motor Deadband).
    """
    # Adiciona/subtrai a força de correção na velocidade de cruzeiro atual
    pwm_l = dynamic_base_speed + correction
    pwm_r = dynamic_base_speed - correction
    
    # --- LÓGICA DE ZONA MORTA (MECÂNICA DE PIVOT) ---
    # Se a roda recebe um sinal fraco (ex: 70), ela apenas consome bateria e trava o robô.
    # Ao jogar esse valor para 0 intencionalmente, forçamos o robô a girar no próprio eixo (pivot),
    # o que resulta em curvas de 90 graus incrivelmente nítidas para a OBR.
    if pwm_l < MIN_MOTION_PWM:
        pwm_l = 0
    if pwm_r < MIN_MOTION_PWM:
        pwm_r = 0
        
    # Assegura matematicamente que o valor não passe do suportado pelo hardware
    pwm_l = int(np.clip(pwm_l, 0, MAX_PWM))
    pwm_r = int(np.clip(pwm_r, 0, MAX_PWM))
    
    # Geração do rótulo da classe para o treinamento da Inteligência Artificial.
    # Aumentamos a margem (20) para classificar oscilações de correção normais como 'Reta' (F).
    if abs(correction) < 20: 
        label = "F"
    else:
        label = "R" if correction > 0 else "L"
        
    return pwm_l, pwm_r, label

def main() -> None:
    # 1. Inicialização de Periféricos
    cap = setup_camera()
    controller = RobotController()

    print("==================================================")
    print(f"[INIT] Sistema de Visão Iniciado.")
    print(f"[INIT] Câmera configurada para: {CAM_FPS} FPS.")
    print(f"[INIT] Área Útil (Deadzone): +/- {int(DEADZONE_PX)} pixels.")
    print("==================================================\n")

    # Flag para ocultar janelas do OpenCV caso esteja rodando diretamente no Pi sem monitor
    HEADLESS_MODE = True 
    last_decision_log = None 
    target_frame_time = 1.0 / CAM_FPS # Tempo alvo por ciclo para manter os FPS cravados

    # Instância do PID configurada empiricamente na pista física
    pid = PIDController(kp=5.0, ki=0.0, kd=0.0)

    try:
        while True:
            loop_start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            # 2. Extração de Features Visuais
            cx_bottom, cx_mid, cx_top, center, thresh = process_vision(frame)

            current_error = 0
            dynamic_base_speed = BASE_SPEED

            # 3. Definição do Alvo (Prioridade inferior)
            if cx_bottom is not None:
                current_error = cx_bottom - center
            elif cx_mid is not None:
                current_error = cx_mid - center

            # 4. Freio Inteligente (Visão Preditiva)
            # Analisa o quão "torta" está a linha comparando a base com o topo da imagem
            if cx_top is not None and cx_bottom is not None:
                curve_intensity = abs(cx_top - cx_bottom)
                if curve_intensity > 40: # Threshold de curva agressiva
                    speed_reduction = min(60, curve_intensity * 0.4) 
                    
                    # O robô freia para entrar suave na curva, MAS nunca abaixo da Zona Morta,
                    # garantindo que ele tenha força mecânica para sair da inércia e fazer o giro.
                    dynamic_base_speed = max(MIN_MOTION_PWM, int(BASE_SPEED - speed_reduction))

            # 5. Aplicação da Zona Morta Visual
            if abs(current_error) < DEADZONE_PX:
                current_error = 0

            # 6. Atuação e Logging
            if cx_bottom is not None or cx_mid is not None:
                # Calcula a atuação, converte para PWM com segurança e envia para o ESP32
                correction = pid.compute(current_error)
                pwm_l, pwm_r, label = calculate_motor_speeds(correction, dynamic_base_speed)
                
                controller.send_pwm(pwm_l, pwm_r)
                controller.save_data(frame, pwm_l, pwm_r, label)
                
                # Previne poluição no terminal, logando apenas mudanças de estado
                if label != last_decision_log:
                    if label == "F":
                        print(f"[{time.strftime('%H:%M:%S')}] [AÇÃO] Linha na área útil. Andando para FRENTE (PWM: {pwm_l}/{pwm_r}).")
                    elif label == "R":
                        print(f"[{time.strftime('%H:%M:%S')}] [AÇÃO] Corrigindo posição para a DIREITA (PWM Esq: {pwm_l}, Dir: {pwm_r}).")
                    elif label == "L":
                        print(f"[{time.strftime('%H:%M:%S')}] [AÇÃO] Corrigindo posição para a ESQUERDA (PWM Esq: {pwm_l}, Dir: {pwm_r}).")
                    last_decision_log = label

            else:
                # 7. Segurança: Sistema à deriva
                pid.integral = 0 # Reseta histórico do PID para não reagir bruscamente depois
                controller.send_pwm(0, 0) # Corta motores instantaneamente
                
                if last_decision_log != "LOST":
                    print(f"[{time.strftime('%H:%M:%S')}] [ALERTA CRÍTICO] Linha perdida! Motores parados por segurança.")
                    last_decision_log = "LOST"

            # ==========================================
            # DEBUG E VISUALIZAÇÃO (Somente se HEADLESS_MODE = False)
            # ==========================================
            if not HEADLESS_MODE:
                cv2.line(frame, (int(center - DEADZONE_PX), 0), (int(center - DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                cv2.line(frame, (int(center + DEADZONE_PX), 0), (int(center + DEADZONE_PX), CAM_HEIGHT), (255, 0, 0), 1)
                
                if cx_bottom is not None: cv2.circle(frame, (cx_bottom, int(CAM_HEIGHT * 0.85)), 8, (0, 255, 0), -1)
                if cx_mid is not None: cv2.circle(frame, (cx_mid, int(CAM_HEIGHT * 0.55)), 8, (0, 255, 255), -1)
                if cx_top is not None: cv2.circle(frame, (cx_top, int(CAM_HEIGHT * 0.25)), 8, (0, 0, 255), -1)

                cv2.imshow("Sistema de Visao - Binarizado", thresh)
                cv2.imshow("Sistema de Visao - RGB", frame)
                
                # Captura tecla 'q' para saída manual
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

            # ==========================================
            # CONTROLE DE CADÊNCIA (THROTTLE DE FPS)
            # ==========================================
            # Força o laço a rodar em exatamente 10 FPS, colocando a thread para dormir o resto do tempo.
            # Isso gera imagens limpas sem arrasto de movimento para a IA.
            processing_time = time.time() - loop_start_time
            if processing_time < target_frame_time:
                time.sleep(target_frame_time - processing_time)

    finally:
        # Bloco finally garante encerramento limpo (motores parados) caso o usuário aperte Ctrl+C
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()