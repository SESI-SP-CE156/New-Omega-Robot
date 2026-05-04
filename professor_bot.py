import cv2
import numpy as np
import serial
import time
import csv
import os

# --- CONFIGURAÇÕES ---
SERIAL_PORT = "/dev/ttyAMA1"  # Porta padrão do ESP32 no Linux
BAUD_RATE = 115200
MAX_PWM = 150
BASE_SPEED = 120  # Velocidade de cruzeiro
ROI_HEIGHT = 0.6  # Analisar os 40% inferiores da imagem

# Configuração de Coleta de Dados
SESSION_DIR = f"data/sessao_{int(time.time())}"
os.makedirs(f"{SESSION_DIR}/images", exist_ok=True)

class RobotController:
    def __init__(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2) # Wait for ESP32 reboot
        except Exception as e:
            print(f"Erro ao conectar no ESP32: {e}")
            self.ser = None

        # Arquivo de Log
        self.log_file = open(f"{SESSION_DIR}/labels.csv", mode='w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["img_path", "pwm_l", "pwm_r", "label"])

    def send_pwm(self, left, right):
        """Envia comandos de velocidade individual para os motores."""
        if self.ser:
            # Protocolo: P,valor_L,valor_R
            command = f"P,{left},{right}\n"
            self.ser.write(command.encode())

    def save_data(self, frame, l_pwm, r_pwm, label):
        timestamp = int(time.time() * 1000)
        img_name = f"img_{timestamp}.jpg"
        img_path = f"images/{img_name}"
        
        cv2.imwrite(f"{SESSION_DIR}/{img_path}", frame)
        self.writer.writerow([img_path, l_pwm, r_pwm, label])

    def close(self):
        self.log_file.close()
        if self.ser: self.ser.close()

def process_vision(frame):
    # 1. Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    # 2. ROI (Region of Interest)
    h, w = thresh.shape
    roi_start = int(h * ROI_HEIGHT)
    roi = thresh[roi_start:h, :]

    # 3. Cálculo do Centro de Massa (Centroid)
    M = cv2.moments(roi)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        return cx, w // 2, frame[roi_start:h, :]
    return None, w // 2, None

def main():
    cap = cv2.VideoCapture(0)
    controller = RobotController()
    
    # Variáveis PID Simples
    kp = 0.8  # Ganho Proporcional

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            cx, center, debug_roi = process_vision(frame)

            if cx is not None:
                # Cálculo do Erro e Ajuste de Trajetória
                error = cx - center
                correction = int(error * kp)

                # Ajuste de PWM Dinâmico
                pwm_l = np.clip(BASE_SPEED + correction, 0, MAX_PWM)
                pwm_r = np.clip(BASE_SPEED - correction, 0, MAX_PWM)

                # Enviar para ESP32
                controller.send_pwm(pwm_l, pwm_r)
                
                # Definir Label para IA (ex: 0=Frente, 1=Esquerda, 2=Direita)
                label = "F" if abs(error) < 20 else ("R" if error > 0 else "L")
                
                # Coleta
                controller.save_data(frame, pwm_l, pwm_r, label)
            else:
                controller.send_pwm(0, 0) # Para se perder a linha

            # Visualização (opcional)
            if cx: cv2.circle(frame, (cx, int(frame.shape[0]*0.8)), 10, (0, 255, 0), -1)
            cv2.imshow("Professor OBR", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        controller.send_pwm(0, 0)
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()