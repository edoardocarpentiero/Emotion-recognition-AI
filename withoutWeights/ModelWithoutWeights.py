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
from utilsModel import (
    focal_loss,
    get_class_distribution,
    plot_distribution,
    balance_class_with_augmentation,
    plot_metrics
)



# Imposta il path del dataset originale e di destinazione
train_dir = "../assets/train"
augmented_dir = "../assets/dataset_bilanciato"

img_size = 48  # più piccolo per una CNN custom
batch_size = 32
epochs = 50
# Target differenziati per classi frequentemente confuse
custom_targets = {
    #'surprise': 27000,
    #'sadness': 27000,
    #'happiness': 27000,
    #'neutral': 26000
}

default_target = 25000  # per tutte le altre classi


os.makedirs(augmented_dir, exist_ok=True)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)



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

for cls in original_dist.keys():
    target_images = custom_targets.get(cls, default_target)
    balance_class_with_augmentation(cls, target_images, train_dir, augmented_dir, datagen)


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
                              min_lr=1e-6)
# Addestramento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[reduce_lr,early_stop]
)

# Salvataggio del modello
model.save("../results/model/cnn_withoutWeights_model.h5")


plot_metrics(history, save_path='../results/plots/withoutWeights/accuracy_and_loss.png')

df = pd.DataFrame(history.history)

# Salva in CSV
df.to_csv("./results/model/training_history_withoutWeights.csv", index=False)

print("✅ Log salvato in training_history_withoutWeights.csv")

