import cv2
import numpy as np
from tensorflow.keras.models import load_model


def webcam(class_names):


    # Carica il modello salvato
    model = load_model("results/model/cnn_model.h5", compile=False)

    # Nomi delle classi FER2013

    # Carica il classificatore Haar Cascade per il rilevamento dei volti
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Acquisisci un frame dalla webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Converti il frame in scala di grigi per il rilevamento del volto
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rileva i volti nel frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Per ogni volto rilevato, disegna un rettangolo e fai la previsione
        for (x, y, w, h) in faces:
            # Disegna un rettangolo attorno al volto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Preprocessa l'immagine del volto
            face_region = gray_frame[y:y + h, x:x + w]  # Seleziona solo la regione del volto
            resized_face = cv2.resize(face_region, (64, 64))  # Ridimensiona l'immagine
            input_face = resized_face.astype('float32') / 255.0  # Normalizza
            input_face = np.expand_dims(input_face, axis=-1)  # Aggiungi il canale
            input_face = np.expand_dims(input_face, axis=0)  # Aggiungi batch dimension

            # Predizione
            prediction = model.predict(input_face)

            # Ottieni la classe con la massima probabilità
            predicted_class = np.argmax(prediction, axis=1)

            # Ottieni la probabilità della classe predetta
            predicted_probability = np.max(prediction)

            # Colorazione dinamica: verde per alta probabilità, rosso per bassa probabilità
            color = (0, 255, 0) if predicted_probability > 0.7 else (0, 0, 255)

            # Mostra il risultato sulla finestra del volto con probabilità colorata
            cv2.putText(frame, f'{class_names[predicted_class[0]]} ({predicted_probability * 100:.2f}%)',
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Visualizza il frame con il volto e le previsioni
        cv2.imshow('Facial Expression Recognition', frame)

        # Interrompi con la pressione di una chiave (ad esempio 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la webcam e chiudi le finestre
    cap.release()
    cv2.destroyAllWindows()
