#!/bin/bash

echo "Configurando Pipeline de Visao - OBR..."

# 1. Configura o Anti-Flicker (Oscilacao de lampadas - 60Hz)
v4l2-ctl -d /dev/video0 -c power_line_frequency=2

# 2. Desativa o Balanço de Branco Automático e fixa a temperatura
v4l2-ctl -d /dev/video0 -c white_balance_automatic=0
v4l2-ctl -d /dev/video0 -c white_balance_temperature=4650

# 3. Desativa a Exposição Automática (Em V4L2, 1 geralmente significa Manual)
v4l2-ctl -d /dev/video0 -c auto_exposure=1

# 4. Fixa o tempo de exposição (Ajuste esse valor na arena!)
# Valores menores = imagem mais escura, porem sem NENHUM borrão de movimento.
# Valores maiores = imagem mais clara, mas pode borrar em alta velocidade.
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=15

# 5. Fixa a saturação e nitidez para valores padrão para evitar processamento extra da câmera
v4l2-ctl -d /dev/video0 -c saturation=128
v4l2-ctl -d /dev/video0 -c sharpness=128

echo "Camera travada e pronta para a pista!"