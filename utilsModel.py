# utilsModel.py

import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss

# Calcola distribuzione per classe
def get_class_distribution(directory):
    return {cls: len(os.listdir(os.path.join(directory, cls)))
            for cls in os.listdir(directory)}

# Mostra distribuzione con grafico
def plot_distribution(dist, title):
    plt.figure(figsize=(10, 5))
    plt.bar(dist.keys(), dist.values(), color='skyblue')
    plt.title(title)
    plt.ylabel("Numero di immagini")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔁 Augment e bilancia
def balance_class_with_augmentation(class_name, target_images, train_dir, augmented_dir, datagen):
    src = os.path.join(train_dir, class_name)
    dst = os.path.join(augmented_dir, class_name)
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.endswith(('.jpg', '.png'))]
    current = len(files)

    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    i = 0
    while current < target_images:
        img_path = os.path.join(src, files[i % len(files)])
        img = load_img(img_path, color_mode='grayscale')
        x = img_to_array(img)
        x = np.expand_dims(x, 0)

        for batch in datagen.flow(x, batch_size=1):
            fname = f"aug_{current}.png"
            fpath = os.path.join(dst, fname)
            plt.imsave(fpath, batch[0].squeeze(), cmap='gray')
            current += 1
            break
        i += 1

# Plot metriche di training
def plot_metrics(history, save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.plot(history.history['loss'], label='Train Loss', color='red', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title('Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
