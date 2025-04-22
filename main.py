from sympy import true, false

import model.ModelTraining
import model.ModelEvaluation

if __name__ == "__main__":

    model.ModelTraining.runTraining('assets/train', 'assets/dataset_bilanciato',8, false)
    model.ModelEvaluation.evaluateModel('assets/test',['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'], false)

    #model.ModelTraining.runTraining('assets/train', 'assets/dataset_bilanciato_fused',5,true)
    #model.ModelEvaluation.evaluateModel('assets/test',['alert', 'disapproval','negative', 'neutral', 'positive'], true)
