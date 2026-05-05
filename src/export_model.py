import tensorflow as tf
import numpy as np
from data_loader import load_ucihar
import os


def rep_dataset(X, n=300):
    for i in range(n):
        yield [X[i].astype(np.float32)[np.newaxis]]


def main():
    X_train, _, _, _, _, _ = load_ucihar()
    model = tf.keras.models.load_model("models/har_baseline_7_target_sit.h5")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_dataset(X_train)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    os.makedirs("models", exist_ok=True)
    with open("models/har_int8_3_target_sit.tflite", "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite INT8 exporté. Taille: {len(tflite_model)/1024:.1f} KB")
    
    # Évaluer la précision TFLite
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    # Détails de quantisation pour l'entrée
    input_details = interpreter.get_input_details()[0]
    in_idx = input_details['index']
    out_idx = interpreter.get_output_details()[0]['index']
    
    scale, zero_point = input_details['quantization'][0], input_details['quantization'][1]
    
    _, _, X_test, y_test, _, _ = load_ucihar()
    correct = 0
    print("⏳ Évaluation du modèle INT8...")
    print("Taille de X_test = ",len(X_test))
    test_len = len(X_test)
    for i in range(test_len):
        # Quantisation manuelle : (float / scale) + zero_point
        input_data = (X_test[i:i+1] / scale + zero_point).astype(np.int8)
        
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(out_idx)
        pred = np.argmax(output_data[0])
        true = np.argmax(y_test[i])
        if pred == true: correct += 1
        
    print(f"📊 Précision TFLite INT8: {correct/test_len:.3f}")

if __name__ == "__main__":
    main()