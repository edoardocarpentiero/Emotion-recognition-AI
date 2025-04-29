import os
import random
import shutil
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

from model.UtilsModel import (
    focal_loss,
    get_class_distribution,
    plot_distribution,
    balance_class_with_augmentation,
    plot_metrics,
    setFusion,
    plot_comparison_distribution,
    split_dataset
)




def runTraining(train_folder, augmented_folder, numberClasses, split_dir_balanced, label_map):
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/model", exist_ok=True)

    img_size = 64
    batch_size = 64
    epochs = 50
    target_images = 25000  # per tutte le classi

    train_dir = train_folder
    augmented_dir = augmented_folder
    os.makedirs(augmented_dir, exist_ok=True)

    # Augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        shear_range=0.1
    )

    # Fusione iniziale delle directory se necessario
    setFusion(train_dir,label_map)
    train_dir = train_dir + "_fused"

    # Analizza distribuzione originale
    original_dist = get_class_distribution(train_dir)
    plot_distribution(original_dist, "Distribuzione prima del bilanciamento", save_path='results/plots/plot_classes_pre_balancing.png')

    # Bilanciamento dataset
    for cls in original_dist.keys():
        print(f"Bilanciamento classe: {cls} ...")
        balance_class_with_augmentation(cls, target_images, train_dir, augmented_dir, datagen)
        print(f"Directory dataset bilanciato: {augmented_dir}/{cls}")

    # Nuova distribuzione
    new_dist = get_class_distribution(augmented_dir)
    plot_comparison_distribution(original_dist, new_dist, "Distribuzione dopo il bilanciamento", save_path='results/plots/plot_classes_post_balancing.png')

    # Divisione in train/val/test
    print("Suddivisione del dataset in train/val/test in "+split_dir_balanced)
    split_dataset(augmented_dir, output_base_dir=split_dir_balanced, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Analizza distribuzione originale
    original_dist = get_class_distribution(split_dir_balanced+'/train')
    plot_distribution(original_dist, "Distribuzione dataset di Training",
                      save_path='results/plots/plot_classes_training.png')

    original_dist = get_class_distribution(split_dir_balanced+'/val')
    plot_distribution(original_dist, "Distribuzione dataset di Validation",
                      save_path='results/plots/plot_classes_validation.png')

    train_generator = train_datagen.flow_from_directory(
        split_dir_balanced+'/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        split_dir_balanced+'/val',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )


    # Modello CNN
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
        Dense(numberClasses, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=focal_loss(),
                  metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-5)

    # Training
    print("Inizio addestramento...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[reduce_lr, early_stop]
    )

    # Salvataggi
    print("Salvataggio modello in results/model/cnn_model.h5")
    model.save("results/model/cnn_model.h5")

    print("Plot metriche e salvataggio history...")
    plot_metrics(history, save_path='results/plots/accuracy_and_loss.png')

    df = pd.DataFrame(history.history)
    df.to_csv("results/model/training_history.csv", index=False)

