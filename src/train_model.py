import tensorflow as tf
from data_loader import load_ucihar
import os
import numpy as np

def augment_har(x, y):
    x = tf.cast(x, tf.float32)
    
    # 1️⃣ Bruit gaussien
    noise = tf.random.normal(tf.shape(x), mean=0.04, stddev=0.05, dtype=tf.float32)
    x = x + noise
    
    # 2️⃣ Variation d'amplitude
    scale = tf.random.uniform([], 0.7, 1.17, dtype=tf.float32)
    x = x * scale
    
    # 3️⃣ Décalage temporel (axis=0 = temps car appliqué AVANT le batch)
    shift = tf.random.uniform([], -4, 4, dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=0)
    
    # 4️⃣ Dropout de canal (broadcast sur la dernière dimension)
    channel_mask = tf.cast(tf.random.uniform([9], dtype=tf.float32) > 0.02, tf.float32)
    x = x * channel_mask

    # 🎯 TILT CIBLÉ : SITTING (3) / STANDING (4)
    # On simule un léger changement de posture sur l'axe Z (body_acc_z, index 2)
    def add_tilt():
        tilt = tf.random.uniform([], -0.17, 0.17, dtype=tf.float32)
        z_tilt = tf.fill([128, 1], tilt)  # Colonne de tilt pour les 128 timesteps
        return tf.concat([x[:, :2], x[:, 2:3] + z_tilt, x[:, 3:]], axis=1)
    
    # y est one-hot → on vérifie si index 3 ou 4 est activé
    is_static = tf.logical_or(tf.greater(y[3], 0.5), tf.greater(y[4], 0.5))
    x = tf.cond(is_static, add_tilt, lambda: x)
    
    return x, y

def build_model(input_shape=(128, 9)):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

def ds_conv_block(x, filters, kernel_size=5):
    x = tf.keras.layers.DepthwiseConv1D(kernel_size, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters, 1, activation='relu')(x)
    return x

def build_model2(input_shape=(128, 9)):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.DepthwiseConv1D(5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(16, 1, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.DepthwiseConv1D(5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 1, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

def ds_conv_block2(x, filters, kernel_size=5, residual=True):
    """Bloc Depthwise Separable + BatchNorm + Residual"""
    # Shortcut pour connexion résiduelle
    if residual:
        # Projection si le nombre de canaux change
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same', use_bias=False)(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = x

    # Depthwise
    x = tf.keras.layers.DepthwiseConv1D(kernel_size, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Pointwise
    x = tf.keras.layers.Conv1D(filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Résidu
    if residual:
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling1D(2)(x)
    return x

def build_model3(input_shape=(128, 9)):
    inputs = tf.keras.Input(shape=input_shape)

    # Stem initial (extraction bas niveau)
    x = tf.keras.layers.Conv1D(32, 5, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = ds_conv_block2(x, 32, residual=False)   # Pas de résidu au début

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

def main():
    X_train, y_train, X_test, y_test, mu, sigma = load_ucihar()
    
    # 💡 Conversion explicite en float32 pour éviter tout conflit dtype
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    model = build_model3()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    model.summary()

    # 🔹 Pipeline CORRECT : shuffle → map (augment) → batch → prefetch
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=39)
    train_ds = train_ds.map(augment_har, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(128)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # 🔹 Validation : pas d'augmentation
    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = val_ds.batch(128).prefetch(tf.data.AUTOTUNE)

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=9, restore_best_weights=True)]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/har_baseline_7_target_sit.h5")
    with open("models/normalization_7_target_sit.json", "w") as f:
        import json; json.dump({"mu": mu, "sigma": sigma}, f)
    print("✅ Modèle & stats sauvegardés.")

if __name__ == "__main__":
    main()