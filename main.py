import os
import shutil
import model.ModelTraining
import model.ModelEvaluation
import model.WebcamDetection

from uitls import (
    cleanFolder
)

train_dir = 'assets/train'
split_dir_balanced = 'assets/dataset_bilanciato_split'
train_dir_balanced = 'assets/dataset_bilanciato_fused'

def menu():
    print("\n--- Menu ---")
    print("1. Costruisci modello all_emotions FER+")
    print("2. Costruisci modello contestualizzato FER+ ")
    print("3. Esci")

def opzione1():
    classes = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    cleanFolder(split_dir_balanced, train_dir_balanced)
    num_classes = len(classes)
    label_map = {
        'anger': 'anger',
        'contempt': 'contempt',
        'disgust': 'disgust',
        'fear': 'fear',
        'surprise': 'surprise',
        'happiness': 'happiness',
        'sadness': 'sadness',
        'neutral': 'neutral'
    }
    print("Esecuzione script...")
    model.ModelTraining.runTraining(train_dir, train_dir_balanced,num_classes,
                                    split_dir_balanced, label_map)
    model.ModelEvaluation.evaluateModel(split_dir_balanced,classes)
    model.WebcamDetection.webcam(classes)


def opzione2():
    classes = ['negative', 'neutral', 'positive']
    cleanFolder(split_dir_balanced,train_dir_balanced)
    num_classes = len(classes)
    print("Esecuzione script...")
    label_map = {
        'contempt': 'negative',
        'sadness': 'negative',
        'happiness': 'positive',
        'neutral': 'neutral'
    }
    model.ModelTraining.runTraining(train_dir, train_dir_balanced, num_classes,   split_dir_balanced, label_map)
    model.ModelEvaluation.evaluateModel(split_dir_balanced, classes)
    model.WebcamDetection.webcam(classes)


def main():
    while True:
        menu()
        scelta = input("Scegli un'opzione: ")

        if scelta == "1":
            opzione1()
        elif scelta == "2":
            opzione2()
        elif scelta == "3":
            print("Uscita dal programma...")
            break
        else:
            print("Opzione non valida, riprova.")


if __name__ == "__main__":
    main()