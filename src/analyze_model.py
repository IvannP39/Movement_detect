"""
Script pour analyser les paramètres et caractéristiques d'un modèle Keras (.h5)
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import argparse
import os


def analyze_model(model_path):
    """
    Charge un modèle .h5 et affiche ses paramètres et caractéristiques
    """
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Le fichier '{model_path}' n'existe pas")
        return
    
    print(f"\n📊 Analyse du modèle: {model_path}\n")
    
    # Charger le modèle
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # ============ INFORMATIONS GÉNÉRALES ============
    print("=" * 80)
    print("📋 INFORMATIONS GÉNÉRALES")
    print("=" * 80)
    print(f"Architecture: {type(model).__name__}")
    print(f"Nombre de couches: {len(model.layers)}")
    print(f"Paramètres entraînables: {model.count_params():,}")
    print(f"État du modèle: {'Compilé' if model.optimizer else 'Non compilé'}")
    if model.optimizer:
        print(f"Optimizer: {model.optimizer.__class__.__name__}")
        print(f"Loss: {model.loss}")
        print(f"Métriques: {model.metrics}")
    
    # ============ FORME D'ENTRÉE/SORTIE ============
    print("\n" + "=" * 80)
    print("🔌 FORME D'ENTRÉE/SORTIE")
    print("=" * 80)
    if hasattr(model, 'input_shape'):
        print(f"Entrée: {model.input_shape}")
    if hasattr(model, 'output_shape'):
        print(f"Sortie: {model.output_shape}")
    
    # ============ TABLEAU DES COUCHES ============
    print("\n" + "=" * 80)
    print("📐 ARCHITECTURE DES COUCHES")
    print("=" * 80)
    
    layers_data = []
    total_params = 0
    
    for i, layer in enumerate(model.layers, 1):
        layer_config = layer.get_config()
        
        # Paramètres de la couche
        layer_params = layer.count_params()
        total_params += layer_params
        
        # Gérer les couches sans output_shape (comme InputLayer)
        try:
            output_shape = str(layer.output_shape)
        except AttributeError:
            # Pour InputLayer et autres couches sans output_shape
            output_shape = str(layer.input_shape) if hasattr(layer, 'input_shape') else 'N/A'
        
        layers_data.append({
            'N°': i,
            'Nom': layer.name,
            'Type': layer.__class__.__name__,
            'Sortie': output_shape,
            'Paramètres': f"{layer_params:,}",
            'Entraînables': 'Oui' if layer.trainable else 'Non'
        })
    
    df_layers = pd.DataFrame(layers_data)
    print(df_layers.to_string(index=False))
    
    # ============ RÉSUMÉ DES PARAMÈTRES ============
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DES PARAMÈTRES PAR TYPE DE COUCHE")
    print("=" * 80)
    
    layer_type_params = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        params = layer.count_params()
        if layer_type not in layer_type_params:
            layer_type_params[layer_type] = 0
        layer_type_params[layer_type] += params
    
    df_params = pd.DataFrame(
        sorted(layer_type_params.items(), key=lambda x: x[1], reverse=True),
        columns=['Type de Couche', 'Nombre de Paramètres']
    )
    df_params['Nombre de Paramètres'] = df_params['Nombre de Paramètres'].apply(lambda x: f"{x:,}")
    print(df_params.to_string(index=False))
    
    # ============ DÉTAILS DES POIDS ============
    print("\n" + "=" * 80)
    print("⚖️  DÉTAILS DES POIDS ET BIAIS")
    print("=" * 80)
    
    weights_data = []
    for layer in model.layers:
        if layer.weights:
            for weight in layer.weights:
                weights_data.append({
                    'Couche': layer.name,
                    'Nom': weight.name,
                    'Forme': str(weight.shape),
                    'Type de Donnée': str(weight.dtype),
                    'Éléments': np.prod(weight.shape)
                })
    
    if weights_data:
        df_weights = pd.DataFrame(weights_data)
        print(df_weights.to_string(index=False))
    else:
        print("Aucun poids trouvé")
    
    # ============ STATISTIQUES DES POIDS ============
    print("\n" + "=" * 80)
    print("📈 STATISTIQUES DES POIDS")
    print("=" * 80)
    
    for layer in model.layers:
        if layer.weights:
            print(f"\n{layer.name}:")
            for weight in layer.weights:
                values = weight.numpy().flatten()
                print(f"  {weight.name}:")
                print(f"    Min: {np.min(values):.6f}")
                print(f"    Max: {np.max(values):.6f}")
                print(f"    Moyenne: {np.mean(values):.6f}")
                print(f"    Écart-type: {np.std(values):.6f}")
    
    # ============ RÉSUMÉ FINAL ============
    print("\n" + "=" * 80)
    print("✅ RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"Total de paramètres: {total_params:,}")
    print(f"Nombre total de couches: {len(model.layers)}")
    
    # Estimation de la taille du modèle
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes par paramètre (float32)
    print(f"Taille estimée (float32): {model_size_mb:.2f} MB")
    
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyser les paramètres d'un modèle Keras (.h5)"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le fichier du modèle (.h5)'
    )
    
    args = parser.parse_args()
    analyze_model(args.model)


if __name__ == "__main__":
    main()
