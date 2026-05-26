#!/bin/bash

echo "=========================================="
echo "    Inicializando Sistema OBR             "
echo "=========================================="

# 1. Verificação e Configuração do Hardware (Câmera)
if [ -e "/dev/video0" ]; then
    echo "[HARDWARE] Câmera detectada em /dev/video0."
    
    # Verifica se o script de câmera existe antes de executar
    if [ -f "./setup_camera.sh" ]; then
        # Utilizamos 'bash' para garantir a execução mesmo se o arquivo não tiver permissão +x
        bash ./setup_camera.sh
    else
        echo "[AVISO] Script setup_camera.sh não encontrado no diretório atual."
    fi
else
    echo "[AVISO] Nenhuma câmera encontrada em /dev/video0. Ignorando v4l2-ctl."
fi

echo "------------------------------------------"

# 2. Verificação e Ativação do Ambiente Virtual
if [ -d "venv" ]; then
    echo "[SOFTWARE] Ativando ambiente OBR (Python 3.10)..."
    source venv/bin/activate
    echo "[SISTEMA] Ambiente pronto! Você já pode rodar o main.py."
else
    echo "[ERRO FATAL] Ambiente virtual 'venv' não encontrado."
    echo "Rode ./setup_venv.sh para criar as dependências primeiro."
fi
echo "=========================================="