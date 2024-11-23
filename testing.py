from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import model_from_json # type: ignore
import numpy
import os
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

def Test_Combine(X,Y):
    json_model = open("models/CNN_v1_Fer2013_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_CNN_1 = model_from_json(loaded_json_model)
    model_CNN_1.load_weights("/Users/hiimbias/PycharmProjects/FED/models/CNN_v1_Fer2013_best_weights.keras")
    # model_CNN_1.summary()
    json_model = open("models/CNN_v2_Fer2013_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_CNN_2 = model_from_json(loaded_json_model)
    model_CNN_2.load_weights("/Users/hiimbias/PycharmProjects/FED/models/CNN_v2_Fer2013_final_weights.keras")
    # model_CNN_2.summary()

    json_model = open("models/ConvSIFTNET_Fer2013_model.json", 'r')
    loaded_json_model = json_model.read()
    json_model.close()
    model_SIFTNET = model_from_json(loaded_json_model)
    model_SIFTNET.load_weights("/Users/hiimbias/PycharmProjects/FED/models/ConvSIFTNET_Fer2013_final_model.keras")

    Split = np.load('dataset/Fer_Usage.npy')
    x_index, = np.where(Split == 'Training')
    y_index, = np.where(Split == 'PublicTest')
    z_index, = np.where(Split == 'PrivateTest')

    X_SIFT = np.load("/Users/hiimbias/PycharmProjects/FED/models/Fer2013_SIFTDetector_Histogram_GEN.npy")
    X_SIFT = X_SIFT.astype('float64')
    print(X_SIFT.shape)

    X_SIFT_Valid = X_SIFT[y_index[0]:y_index[-1]+1]
    # X_SIFT_Test = X_SIFT[z_index[0]:z_index[-1]+1]
    X_SIFT_Test = X_SIFT[:len(X)]

    predicted_V1 = model_CNN_1.predict(X)
    predicted_V2 = model_CNN_2.predict(X)
    predicted_SIFT = model_SIFTNET.predict([X, X_SIFT_Test])
    predicted_combine =    (predicted_SIFT + predicted_V1+predicted_V2)/3.0



    True_Y = []
    Predicted_Y = []
    predicted_list = predicted_combine.tolist()
    true_Y_list = Y.tolist()

    for i in range(len(Y)):
        Proba_max = max(predicted_combine[i])
        current_class = max(true_Y_list[i])
        class_of_Predict_Y = predicted_list[i].index(Proba_max)
        class_of_True_Y = true_Y_list[i].index(current_class)

        True_Y.append(class_of_True_Y)
        Predicted_Y.append(class_of_Predict_Y)

    print("Accuracy on test set :" + str(accuracy_score(True_Y,Predicted_Y)*100) + "%")
    print("Confusion Matrix:\n" + str(confusion_matrix(True_Y,Predicted_Y)))
    print("Classification Report:\n" + str(classification_report(True_Y,Predicted_Y)))

    np.save("Fer2013_True_y_SIFT", True_Y)
    np.save("Fer2013_Predict_y_SIFT",Predicted_Y)

Y = np.load("dataset/Fer2013_Y_test.npy")
X = np.load("dataset/Fer2013_X_test.npy")
Test_Combine(X,Y)










