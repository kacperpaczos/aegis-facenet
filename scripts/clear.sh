#!/bin/bash

# Przejście do katalogu głównego projektu
cd ..

# Sprawdzenie czy venv istnieje
if [ -d "venv" ]; then
    echo "Usuwanie środowiska wirtualnego..."
    rm -rf venv
    echo "Środowisko wirtualne zostało usunięte"
else
    echo "Środowisko wirtualne nie istnieje"
fi

# Usuwanie plików cache Pythona
echo "Usuwanie plików cache..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

echo "Czyszczenie zakończone pomyślnie!" 