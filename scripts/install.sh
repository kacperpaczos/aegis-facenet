#!/bin/bash

# Aktualizacja systemu
echo "Aktualizacja systemu..."
sudo apt-get update
sudo apt-get upgrade -y

# Instalacja wymaganych pakietów systemowych
echo "Instalacja wymaganych pakietów..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-ensurepip \
    libgl1-mesa-glx \
    libglib2.0-0

# Przejście do katalogu głównego projektu
cd ..

# Tworzenie wirtualnego środowiska
echo "Tworzenie wirtualnego środowiska Python..."
python3 -m venv venv
source venv/bin/activate

# Aktualizacja pip
echo "Aktualizacja pip..."
pip install --upgrade pip

# Instalacja wymaganych pakietów Pythona
echo "Instalacja zależności Pythona..."
pip install \
    facenet-pytorch \
    torch \
    torchvision \
    Pillow \
    numpy \
    opencv-python \
    aiohttp

echo "Instalacja zakończona pomyślnie!" 