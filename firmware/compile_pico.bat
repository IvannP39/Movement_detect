@echo off
REM Script de compilation pour Raspberry Pi Pico W (Windows)

echo Compilation du firmware pour Pico W...
cd /d %~dp0

arduino-cli compile --fqbn rp2040:rp2040:rpipico2w --build-path .\build_pico --libraries .\libraries .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Compilation reussie!
    echo.
    echo Pour uploader sur Pico W:
    echo 1. Branchez le Pico W en maintenant BOOTSEL appuye
    echo 2. Attendez que le lecteur CIRCUITPY apparaisse
    echo 3. Copiez le fichier: build_pico\firmware.ino.uf2 sur CIRCUITPY
    echo.
    echo Le firmware s'installera automatiquement
) else (
    echo.
    echo [ERREUR] Compilation echouee
    exit /b 1
)

pause
