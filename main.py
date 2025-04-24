from sympy import true, false

import model.ModelTraining
import model.ModelEvaluation
import model.WebcamDetection

if __name__ == "__main__":


    #model.ModelTraining.runTraining('assets/train', 'assets/dataset_bilanciato_fused',4)
    #model.ModelEvaluation.evaluateModel('assets/test',['ambiguous', 'negative', 'neutral', 'positive'])

   model.WebcamDetection.webcam(['ambiguous', 'negative', 'neutral', 'positive'])
