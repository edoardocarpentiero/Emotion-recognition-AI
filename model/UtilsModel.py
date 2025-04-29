# UtilsModel.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import random
import os
import shutil
from PIL import Image
label_map = {}

def split_dataset(source_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Divide le immagini da source_dir in train/val/test dentro output_base_dir.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Le proporzioni devono sommare a 1."

    classes = os.listdir(source_dir)

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        train_split = int(train_ratio * len(images))
        val_split = int((train_ratio + val_ratio) * len(images))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            split_dir = os.path.join(output_base_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(split_dir, img)
                shutil.copyfile(src, dst)


def setFusion(datasetFolder, labelMap):
    label_map = labelMap
    src_root = datasetFolder
    dst_root = datasetFolder + "_fused"

    os.makedirs(dst_root, exist_ok=True)

    for old_label in os.listdir(src_root):
        old_path = os.path.join(src_root, old_label)
        if not os.path.isdir(old_path):
            continue

        new_label = label_map.get(old_label)
        if new_label is None:
            continue

        dst_path = os.path.join(dst_root, new_label)
        os.makedirs(dst_path, exist_ok=True)

        for fname in os.listdir(old_path):
            src_img = os.path.join(old_path, fname)
            dst_img = os.path.join(dst_path, fname)

            try:
                # Apri e ridimensiona l'immagine a 64x64
                img = Image.open(src_img).convert("L")  # Grayscale
                img = img.resize((64, 64))
                img.save(dst_img)
            except Exception as e:
                print(f"Errore con {src_img}: {e}")


# Focal loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=-1)

    return focal_loss_fixed

# Calcola distribuzione per classe
def get_class_distribution(directory):
    return {cls: len(os.listdir(os.path.join(directory, cls)))
            for cls in os.listdir(directory)}

# Mostra distribuzione con grafico
def plot_distribution(dist, title, save_path=None):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(dist.keys(), dist.values(), color='skyblue')
    plt.title(title)
    plt.ylabel("Numero di immagini")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Aggiunta dei numeri sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_comparison_distribution(dist_before, dist_after, title, save_path=None):
    labels = list(dist_before.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, dist_before.values(), width, label='Prima del bilanciamento', color='#87CEEB', alpha=0.7)
    bars2 = ax.bar(x + width/2, dist_after.values(), width, label='Dopo il bilanciamento', color='#408FBD', alpha=0.7)

    # Etichette numeriche sopra le barre
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    ax.set_title(title)
    ax.set_ylabel('Numero di immagini')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
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

