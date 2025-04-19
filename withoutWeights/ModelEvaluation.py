import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam

def prepare_confusion_summary(errors_dict, class_names, top_n=10):
    confusion_df = pd.DataFrame(0, index=class_names, columns=class_names)

    for key, count in errors_dict.items():
        true_label, pred_label = key.split(" → ")
        confusion_df.loc[true_label, pred_label] = count

    # Mostra solo le top N confusioni
    flattened = confusion_df.stack().sort_values(ascending=False)
    top_confusions = flattened[:top_n]

    return top_confusions.unstack().fillna(0)

# === Impostazioni base ===
model = load_model("../results/model/cnn_model.h5", compile=False)
test_dir = '../assets/test'
img_size = 48
batch_size = 32
class_names = ['anger', 'contempt','disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

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
    optimizer='adam',
    loss='categorical_crossentropy',  # o 'sparse_categorical_crossentropy' a seconda dei tuoi label
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
os.makedirs("../results/plots/withoutWeights", exist_ok=True)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix With Weights')
plt.tight_layout()
plt.savefig('../results/plots/withoutWeights/confusion_matrix.png', dpi=300)
plt.show()

# === Classification Report ===
report = classification_report(true_labels, predicted_classes, target_names=class_names)
print("Classification Report:")
print(report)

os.makedirs("../results/report/withoutWeights", exist_ok=True)
with open('../results/report/withoutWeights/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# === Analisi classi confuse ===
errors = {}
for true, pred in zip(true_labels, predicted_classes):
    if true != pred:
        true_class = class_names[true]
        pred_class = class_names[pred]
        key = f"{true_class} → {pred_class}"
        errors[key] = errors.get(key, 0) + 1

summary_df = prepare_confusion_summary(errors, class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(summary_df, annot=True, fmt='g', cmap='Reds')
plt.title("Top confusioni di classe")
plt.ylabel("Classe vera")
plt.xlabel("Classe predetta")
plt.tight_layout()
plt.savefig('../results/plots/withoutWeights/top_confused_classes_heatmap.png', dpi=300)
plt.show()
sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)

print("\n🔍 Classi più frequentemente confuse:")
for err, count in sorted_errors[:10]:
    print(f"{err}: {count} volte")

with open('../results/report/withoutWeights/top_confused_classes.txt', 'w', encoding='utf-8') as f:
    for err, count in sorted_errors:
        f.write(f"{err}: {count} volte\n")

# === Salva le immagini sbagliate in cartelle separate ===
os.makedirs("../results/errors/withoutWeights", exist_ok=True)

for i, (true, pred) in enumerate(zip(true_labels, predicted_classes)):
    if true != pred:
        # Percorso dell'immagine sbagliata
        img_path = test_generator.filepaths[i]
        img = load_img(img_path, color_mode='grayscale',
                       target_size=(img_size, img_size))  # Carica come immagine in scala di grigio

        true_class = class_names[true]
        pred_class = class_names[pred]

        # Crea una cartella per gli errori specifici
        error_dir = os.path.join("../results/errors/withoutWeights", f"{true_class}_pred_{pred_class}")
        os.makedirs(error_dir, exist_ok=True)

        # Salva l'immagine sbagliata
        img.save(os.path.join(error_dir, os.path.basename(img_path)))

        # Puoi anche salvarla come numpy array se preferisci
        # np.save(os.path.join(error_dir, os.path.basename(img_path).replace('.jpg', '.npy')), img_array)



