#!/bin/bash
# Script de compilation pour Raspberry Pi Pico W

echo "Compiling firmware for Pico W..."
arduino-cli compile --fqbn rp2040:rp2040:rpipico2w --build-path ./build_pico --libraries ./libraries .

if [ $? -eq 0 ]; then
    echo "✓ Compilation réussie!"
    echo "Le firmware est prêt à être uploadé."
    echo ""
    echo "Pour uploader sur Pico W:"
    echo "1. Branchez le Pico W en maintenant BOOTSEL appuyé"
    echo "2. Lancez: arduino-cli upload --fqbn rp2040:rp2040:rpipico2w --input-file ./build_pico/firmware.ino.elf"
    echo "   OU"
    echo "2. Copiez le fichier build_pico/firmware.ino.uf2 sur le lecteur de masse CIRCUITPY"
else
    echo "✗ Compilation échouée"
    exit 1
fi
