#!/bin/bash

echo "--- Iniciando configuração do ambiente OBR ---"

# Verifica se o Python 3.10 está instalado
if ! command -v python3.10 &> /dev/null
then
    echo "Erro: Python 3.10 não encontrado. Por favor, instale-o antes de continuar."
    exit
fi

# Cria o ambiente virtual chamado 'venv'
python3.10 -m venv venv
echo "✅ Ambiente virtual criado."

# Ativa temporariamente para instalar as dependências
source venv/bin/activate

# Atualiza o pip e instala os requisitos
echo "--- Instalando dependências ---"
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Tudo pronto! Use './enter_venv.sh' para começar a trabalhar."
deactivate