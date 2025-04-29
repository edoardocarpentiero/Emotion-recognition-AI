import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from model.UtilsModel import (
    focal_loss,
    get_class_distribution,
    plot_distribution
)


def evaluateModel(datasetTestFolder, class_names):
    test_dir = datasetTestFolder+'/test'

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/errors", exist_ok=True)
    os.makedirs("results/report", exist_ok=True)

    # === Impostazioni base ===
    model = load_model("results/model/cnn_model.h5", compile=False)

    img_size = 64
    batch_size = 64

    original_dist = get_class_distribution(test_dir)
    plot_distribution(original_dist, "Distribuzione classi di Test",
                      save_path='results/plots/plot_classes_test.png')

    # === Caricamento dati di test ===
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale'
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    model.evaluate(test_generator, verbose=1)


    # === Predizioni ===
    predictions = model.predict(test_generator, verbose=1)
    true_labels = test_generator.classes
    predicted_classes = np.argmax(predictions, axis=1)

    # === Confusion Matrix ===
    conf_matrix = confusion_matrix(true_labels, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)


    # === Plot confusion matrix ===

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.ylabel("Classe vera")
    plt.xlabel("Classe predetta")
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrix.png', dpi=300)
    plt.show()

    # === Classification Report ===
    report = classification_report(true_labels, predicted_classes, target_names=class_names)
    print("Classification Report:")
    print(report)

    with open('results/report/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    analysisImagesConfused(true_labels, predicted_classes, class_names, test_generator, img_size)



# === Analisi classi confuse ===
def analysisImagesConfused(true_labels, predicted_classes, class_names, test_generator, img_size):
    errors = {}
    for true, pred in zip(true_labels, predicted_classes):
        if true != pred:
            true_class = class_names[true]
            pred_class = class_names[pred]
            key = f"{true_class} → {pred_class}"
            errors[key] = errors.get(key, 0) + 1


    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)

    print("\n🔍 Classi più frequentemente confuse:")
    for err, count in sorted_errors[:10]:
        print(f"{err}: {count} volte")

    with open('results/report/top_confused_classes.txt', 'w', encoding='utf-8') as f:
        for err, count in sorted_errors:
            f.write(f"{err}: {count} volte\n")


    for i, (true, pred) in enumerate(zip(true_labels, predicted_classes)):
        if true != pred:
            # Percorso dell'immagine sbagliata
            img_path = test_generator.filepaths[i]
            img = load_img(img_path, color_mode='grayscale',
                           target_size=(img_size, img_size))  # Carica come immagine in scala di grigio

            true_class = class_names[true]
            pred_class = class_names[pred]

            # Crea una cartella per gli errori specifici
            error_dir = os.path.join("results/errors", f"{true_class}_pred_{pred_class}")
            os.makedirs(error_dir, exist_ok=True)

            # Salva l'immagine sbagliata
            img.save(os.path.join(error_dir, os.path.basename(img_path)))




