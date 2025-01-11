#!/bin/bash

# Sprawdzenie czy venv istnieje
if [ ! -d "../venv" ]; then
    echo "Błąd: Nie znaleziono środowiska wirtualnego (venv)!"
    echo "Uruchom najpierw ./install.sh"
    exit 1
fi

# Przejście do katalogu głównego projektu
cd ..

# Określenie ścieżki do aktywacji venv
VENV_ACTIVATE="venv/bin/activate"

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Błąd: Nie można znaleźć pliku aktywacyjnego venv!"
    exit 1
fi

# Aktywacja wirtualnego środowiska
echo "Aktywacja środowiska wirtualnego..."
source "$VENV_ACTIVATE"

# Sprawdzenie czy aktywacja się powiodła
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Błąd: Nie udało się aktywować środowiska wirtualnego!"
    exit 1
fi

echo "Środowisko wirtualne aktywowane pomyślnie"

# Uruchomienie serwera
echo "Uruchamianie serwera Facenet..."
python aegis.facenet.py

# W przypadku błędu, wyświetl komunikat
if [ $? -ne 0 ]; then
    echo "Wystąpił błąd podczas uruchamiania serwera!"
    exit 1
fi 