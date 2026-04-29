#!/bin/bash

# Verifica se o diretório do venv existe
if [ -d "venv" ]; then
    echo "--- Ativando ambiente OBR (Python 3.10) ---"
    source venv/bin/activate
else
    echo "Erro: Ambiente virtual 'venv' não encontrado. Rode ./setup_venv.sh primeiro."
fi