import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.models import Sequential
import pickle
from keras.regularizers import l2
import tensorflow as tf
import pandas as pd

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss


# Imposta il path del dataset originale e di destinazione
train_dir = "assets/train"
augmented_dir = "assets/dataset_bilanciato"

img_size = 48  # più piccolo per una CNN custom
batch_size = 32
epochs = 50
# Target differenziati per classi frequentemente confuse
custom_targets = {
    'surprise': 27000,
    'sadness': 27000,
    'happiness': 27000,
    'neutral': 26000
}

default_target = 25000  # per tutte le altre classi


os.makedirs(augmented_dir, exist_ok=True)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Visualizza la distribuzione iniziale
def get_class_distribution(directory):
    return {cls: len(os.listdir(os.path.join(directory, cls)))
            for cls in os.listdir(directory)}

def plot_distribution(dist, title):
    plt.figure(figsize=(10, 5))
    plt.bar(dist.keys(), dist.values(), color='skyblue')
    plt.title(title)
    plt.ylabel("Numero di immagini")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

original_dist = get_class_distribution(train_dir)
plot_distribution(original_dist, "Distribuzione prima del bilanciamento")


# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Funzione per copiare immagini e generare quelle mancanti
def balance_class_with_augmentation(class_name, target_images):
    src = os.path.join(train_dir, class_name)
    dst = os.path.join(augmented_dir, class_name)
    os.makedirs(dst, exist_ok=True)

    files = [f for f in os.listdir(src) if f.endswith(('.jpg', '.png'))]
    current = len(files)

    # Copia immagini esistenti
    for f in files:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    i = 0
    while current < target_images:
        img_path = os.path.join(src, files[i % len(files)])  # Ciclare sulle immagini
        img = load_img(img_path, color_mode='grayscale')  # Grayscale
        x = img_to_array(img)
        x = np.expand_dims(x, 0)

        for batch in datagen.flow(x, batch_size=1):
            fname = f"aug_{current}.png"
            fpath = os.path.join(dst, fname)
            plt.imsave(fpath, batch[0].squeeze(), cmap='gray')
            current += 1
            break
        i += 1

for cls in original_dist.keys():
    target_images = custom_targets.get(cls, default_target)
    balance_class_with_augmentation(cls, target_images)


# Visualizza distribuzione dopo il bilanciamento
new_dist = get_class_distribution(augmented_dir)
plot_distribution(new_dist, "Distribuzione dopo il bilanciamento")


train_generator = train_datagen.flow_from_directory(
    augmented_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale',  # Carica come immagini in scala di grigio (1 canale)
)

val_generator = train_datagen.flow_from_directory(
    augmented_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale',  # Carica come immagini in scala di grigio (1 canale)
)

weight_decay = 1e-4

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay), input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(weight_decay)),
    Dropout(0.3),
    Dense(8, activation='softmax')  # 8 classi
])

# 🔧 Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=focal_loss(gamma=2.0, alpha=0.25),
              metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',   # Monitora la loss di validazione
                              factor=0.1,           # Riduce il learning rate del 10%
                              patience=2,           # Attende 3 epoche senza miglioramenti prima di ridurre
                              verbose=1,            # Stampa un messaggio quando il learning rate viene ridotto
                              min_lr=1e-6)
# Addestramento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[reduce_lr,early_stop]
)

# Salvataggio del modello
model.save("model/cnn_model.h5")

# Grafico accuracy & loss
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')

    # Loss
    plt.plot(history.history['loss'], label='Train Loss', color='red', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--')

    plt.title('Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/accuracy_and_loss.png', dpi=300)
    plt.show()

plot_metrics(history)

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

