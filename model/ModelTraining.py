import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.models import Sequential
from keras.regularizers import l2
import pandas as pd

from model.utilsModel import (
    focal_loss,
    get_class_distribution,
    plot_distribution,
    balance_class_with_augmentation,
    plot_metrics,
    setFusion
)



def runTraining(train_folder, augmented_folder, numberClasses, withFusion):
    # Imposta il path del dataset originale e di destinazione
    folder = ""

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/model", exist_ok=True)

    train_dir = train_folder
    augmented_dir = augmented_folder

    img_size = 48
    batch_size = 32
    epochs = 50

    target_images = 25000  # per tutte le altre classi


    os.makedirs(augmented_dir, exist_ok=True)

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
    if (withFusion):
        folder = "withFusion"
        setFusion(train_dir)
    else:
        folder = "withoutFusion"

    original_dist = get_class_distribution(train_dir)
    plot_distribution(original_dist, "Distribuzione prima del bilanciamento")

    for cls in original_dist.keys():
        balance_class_with_augmentation(cls, target_images, train_dir, augmented_dir, datagen)


    # Visualizza distribuzione dopo il bilanciamento
    new_dist = get_class_distribution(augmented_dir)
    plot_distribution(new_dist, "Distribuzione dopo il bilanciamento")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

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
        Dense(numberClasses, activation='softmax')
    ])

    # 🔧 Compile
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=focal_loss(gamma=2.0, alpha=0.25),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',   # Monitora la loss di validazione
                                  factor=0.1,           # Riduce il learning rate del 10%
                                  patience=1,           # Attende 3 epoche senza miglioramenti prima di ridurre
                                  min_lr=1e-5)
    # Addestramento
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[reduce_lr,early_stop]
    )

    # Salvataggio del modello
    model.save("results/model/cnn_"+folder+"_model.h5")


    plot_metrics(history, save_path='results/plots/accuracy_and_loss_'+folder+'.png')

    df = pd.DataFrame(history.history)

    # Salva in CSV
    df.to_csv("results/model/training_history_"+folder+".csv", index=False)

