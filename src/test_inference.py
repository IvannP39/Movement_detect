#!/usr/bin/env python3
"""
Test d'inférence pour le modèle HAR TinyML
- Charge le modèle (Keras ou TFLite)
- Prédit sur un échantillon du test set ou sur des données custom
- Affiche : label prédit, confiance, comparaison avec vérité terrain
- Optionnel : visualisation matplotlib des signaux + barres de confiance
"""

import os
import sys
import json
import numpy as np
import argparse
import sys

# Force UTF-8 on Windows for emoji/special char support
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Imports conditionnels ─────────────────────────────────────
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Constants ─────────────────────────────────────────────────
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
          "SITTING", "STANDING", "LAYING"]
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z"
]

# ── Helpers ───────────────────────────────────────────────────
def load_normalization(path="models/normalization_7_target_sit.json"):
    with open(path, "r") as f:
        stats = json.load(f)
    return np.array(stats["mu"]), np.array(stats["sigma"])

def load_test_sample(X_test, y_test, idx=0):
    """Retourne un échantillon prétraité + son label vrai"""
    return X_test[idx][np.newaxis], y_test[idx], idx

def normalize_sample(sample, mu, sigma):
    """Applique la même normalisation qu'à l'entraînement"""
    return (sample - mu) / sigma

def predict_keras(model, sample):
    pred = model.predict(sample, verbose=0)[0]
    return pred, np.argmax(pred)

def predict_tflite(path, sample):
    """
    Exécute une inférence TFLite en gérant automatiquement la quantification
    si le modèle est de type INT8.
    """
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_idx = input_details['index']
    out_idx = output_details['index']

    # 1. Gestion de la quantification d'entrée
    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        # Quantification : (float - zero_point) * scale ? 
        # Non, la formule inverse pour passer de float à int8 est : (float / scale) + zero_point
        input_data = (sample / scale + zero_point).astype(np.int8)
    else:
        input_data = sample.astype(np.float32)

    interpreter.set_tensor(in_idx, input_data)
    interpreter.invoke()

    # 2. Récupération du résultat et déquantification si nécessaire
    output_data = interpreter.get_tensor(out_idx)

    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        # Déquantification : (int8 - zero_point) * scale
        output_data = (output_data.astype(np.float32) - zero_point) * scale

    pred = output_data[0]
    return pred, np.argmax(pred)

def plot_sample(sample, pred_probs, true_label, pred_label):
    """Visualisation : 9 canaux + barre de confiance"""
    if not HAS_MPL:
        print("[WARN] matplotlib non installé -> skip visualisation")
        return
    
    fig = plt.figure(figsize=(14, 6))
    
    # Partie 1 : signaux bruts (9 canaux)
    ax1 = plt.subplot(1, 2, 1)
    x_axis = np.arange(128)
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    for i in range(9):
        ax1.plot(x_axis, sample[0, :, i], label=SIGNALS[i], color=colors[i], linewidth=0.8)
    ax1.set_title("Signals d'entrée (fenêtre 128 samples)")
    ax1.set_xlabel("Échantillon")
    ax1.set_ylabel("Valeur normalisée")
    ax1.legend(fontsize=7, ncol=3)
    ax1.grid(alpha=0.3)
    
    # Partie 2 : barres de confiance
    ax2 = plt.subplot(1, 2, 2)
    y_pos = np.arange(6)
    bars = ax2.barh(y_pos, pred_probs, color=['#2ecc71' if i==pred_label else '#95a5a6' for i in range(6)])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(LABELS)
    ax2.set_xlabel("Confiance")
    ax2.set_title("Predictions")
    
    # Marqueur vérité terrain
    if true_label is not None:
        ax2.axhline(true_label, color='#e74c3c', linestyle='--', linewidth=1, label='Vérité terrain')
        ax2.legend()
    
    # Annotation résultat
    status = "OK" if pred_label == true_label else "WRONG"
    plt.suptitle(f"Result : {LABELS[pred_label]} | {status}", fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()

def print_prediction(pred_probs, pred_label, true_label=None):
    """Affichage texte dans le terminal"""
    print("\n" + "="*60)
    print(f"PREDICTION : {LABELS[pred_label]}")
    print(f"Confidence : {pred_probs[pred_label]*100:.1f}%")
    if true_label is not None:
        status = "OK" if pred_label == true_label else "WRONG"
        print(f"Ground Truth : {LABELS[true_label]} | {status}")
    print("\nConfidence details :")
    for i, (label, prob) in enumerate(zip(LABELS, pred_probs)):
        bar = "█" * int(prob * 40)
        print(f"  {label:20s} {bar} {prob*100:5.1f}%")
    print("="*60 + "\n")

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test d'inférence HAR TinyML")
    parser.add_argument("--model", type=str, required=True, 
                        help="Chemin vers le modèle (.h5 ou .tflite)")
    parser.add_argument("--idx", type=int, default=0, 
                        help="Index de l'échantillon de test à prédire")
    parser.add_argument("--no-plot", action="store_true", 
                        help="Désactiver l'affichage matplotlib")
    parser.add_argument("--custom", type=str, default=None,
                        help="Fichier .npy avec un échantillon custom (shape: 128x9)")
    args = parser.parse_args()

    # ── Charger normalisation ─────────────────────────────
    mu, sigma = load_normalization()
    mu = mu.reshape(1, 1, -1)
    sigma = sigma.reshape(1, 1, -1)

    # ── Charger ou générer l'entrée ────────────────────────
    if args.custom:
        print(f"Loading custom data : {args.custom}")
        sample = np.load(args.custom)  # Doit être (128, 9) ou (1, 128, 9)
        if sample.ndim == 2:
            sample = sample[np.newaxis]  # → (1, 128, 9)
        sample = normalize_sample(sample, mu, sigma)
        y_true = None
        print("[WARN] Mode custom : pas de vérité terrain disponible")
    else:
        print("Loading UCI HAR test set...")
        from data_loader import load_ucihar
        _, _, X_test, y_test, _, _ = load_ucihar()
        sample, y_true, idx = load_test_sample(X_test, y_test, args.idx)
        print(sample)
        #sample = normalize_sample(sample, mu, sigma)
        print(f"Test sample #{idx} selected")

    # ── Charger modèle et prédire ──────────────────────────
    print(f"Loading model : {args.model}")
    
    if args.model.endswith(".h5"):
        if not HAS_TF:
            print("[ERROR] TensorFlow required for .h5 models")
            sys.exit(1)
        model = tf.keras.models.load_model(args.model)
        pred_probs, pred_label = predict_keras(model, sample)
        
    elif args.model.endswith(".tflite"):
        if not HAS_TF:
            print("[ERROR] TensorFlow required for .tflite models")
            sys.exit(1)
        pred_probs, pred_label = predict_tflite(args.model, sample)
        
    else:
        print("[ERROR] Unsupported model extension (.h5 or .tflite required)")
        sys.exit(1)

    # ── Afficher résultats ─────────────────────────────────
    y_true_idx = np.argmax(y_true) if y_true is not None else None
    print_prediction(pred_probs, pred_label, y_true_idx)
    
    if not args.no_plot and HAS_MPL:
        plot_sample(sample, pred_probs, y_true_idx, pred_label)
    elif not args.no_plot and not HAS_MPL:
        print("Tip : install matplotlib for visualization -> pip install matplotlib")

if __name__ == "__main__":
    main()