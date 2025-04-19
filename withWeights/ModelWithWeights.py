import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from utilsModel import (
    focal_loss,
    get_class_distribution,
    plot_distribution,
    balance_class_with_augmentation,
    plot_metrics
)



# 📂 Paths
train_dir = "../assets/train"
augmented_dir = "../assets/dataset_bilanciato"
csv_path = "../assets/fer2013new.csv"

img_size = 48
batch_size = 32
epochs = 50

# 📊 Target augmentation
target_images = 25000

# 📁 Carica CSV e calcola pesi per classe
df = pd.read_csv(csv_path)
df = df[(df["Usage"].isin(["Training", "PublicTest"])) & (df["Image name"].notna())]

emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
emotion_votes = {emotion: 0 for emotion in emotion_cols}

for _, row in df.iterrows():
    for emotion in emotion_cols:
        emotion_votes[emotion] += row[emotion]

total_votes = sum(emotion_votes.values())
class_freq = {k: v / total_votes for k, v in emotion_votes.items()}
class_weight_dict = {i: 1.0 / freq for i, (k, freq) in enumerate(class_freq.items())}
max_weight = max(class_weight_dict.values())
class_weight_dict = {k: v / max_weight for k, v in class_weight_dict.items()}

print("📈 Class Weights:", class_weight_dict)

# 🔄 ImageDataGenerator
os.makedirs(augmented_dir, exist_ok=True)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 📈 Distribuzione classi
original_dist = get_class_distribution(train_dir)
plot_distribution(original_dist, "Distribuzione prima del bilanciamento")

# 📸 Augmentation settings
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
    balance_class_with_augmentation(cls, target_images, train_dir, augmented_dir, datagen)


# 📈 Distribuzione dopo il bilanciamento
new_dist = get_class_distribution(augmented_dir)
plot_distribution(new_dist, "Distribuzione dopo il bilanciamento")

# 🔄 Generatori
train_generator = train_datagen.flow_from_directory(
    augmented_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale',
)

val_generator = train_datagen.flow_from_directory(
    augmented_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale',
)

# 🧩 Allinea class_weight con ordine classi
print("Class index mapping:", train_generator.class_indices)
index_to_class = {v: k for k, v in train_generator.class_indices.items()}
ordered_weights = {}

for i in range(len(index_to_class)):
    emotion_name = index_to_class[i]
    if emotion_name in emotion_cols:
        idx_in_emotion = emotion_cols.index(emotion_name)
        ordered_weights[i] = class_weight_dict[idx_in_emotion]

print("🔧 Ordered class weights:", ordered_weights)

# 🧠 Modello CNN
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
    Dense(8, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=focal_loss(gamma=2.0, alpha=0.25),
              metrics=['accuracy'])

# 🛑 Early stopping & riduzione LR
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    min_lr=1e-6
)

# 🚀 Addestramento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[reduce_lr, early_stop],
    class_weight=ordered_weights
)

# 💾 Salvataggio
model.save("../results/model/cnn_withWeights_model.h5")

# 📊 Grafici
plot_metrics(history, save_path='../results/plots/witWeights/accuracy_and_loss.png')

df = pd.DataFrame(history.history)

# Salva in CSV
df.to_csv("./results/model/training_history_withWeights.csv", index=False)

print("✅ Log salvato in training_history_withWeights.csv")


