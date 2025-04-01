from time import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import model_from_json

def Test_Accuracy_And_Speed(X, Y):
    print("Loading models...")
    # Load models
    with open("models/CNN_v1/CNN_v1_Fer2013_model.json", 'r') as json_file:
        model_CNN_1 = model_from_json(json_file.read())
    model_CNN_1.load_weights("/Users/hiimbias/PycharmProjects/FED/models/CNN_v1_Fer2013_best_weights.keras")

    with open("models/CNN_v2/CNN_v2_Fer2013_model.json", 'r') as json_file:
        model_CNN_2 = model_from_json(json_file.read())
    model_CNN_2.load_weights("/Users/hiimbias/PycharmProjects/FED/models/CNN_v2_Fer2013_final_weights.keras")

    with open("models/CNN_SIFT/ConvSIFTNET_Fer2013_model.json", 'r') as json_file:
        model_SIFTNET = model_from_json(json_file.read())
    model_SIFTNET.load_weights("/Users/hiimbias/PycharmProjects/FED/models/ConvSIFTNET_Fer2013_final_model.keras")

    # Load SIFT data
    X_SIFT = np.load("/models/descriptors/Fer2013_SIFTDetector_Histogram_GEN.npy").astype('float64')
    X_SIFT_Test = X_SIFT[:len(X)] # Use only the first len(X) samples
    print(X_SIFT.shape)

    # Evaluate models
    model_results = []

    def evaluate_model(model, X_data, Y_data, model_name, is_sift=False):
        start_time = time()
        if is_sift:
            predictions = model.predict([X_data, X_SIFT_Test])
        else:
            predictions = model.predict(X_data)
        end_time = time()
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(Y_data, axis=1)
        acc = accuracy_score(true_classes, predicted_classes)
        processing_time = end_time - start_time
        model_results.append((model_name, acc * 100, processing_time))
        return predictions

    # Test individual models
    print("Testing CNN_v1...")
    predictions_v1 = evaluate_model(model_CNN_1, X, Y, "CNN_v1")

    print("Testing CNN_v2...")
    predictions_v2 = evaluate_model(model_CNN_2, X, Y, "CNN_v2")

    print("Testing ConvSIFTNET...")
    predictions_sift = evaluate_model(model_SIFTNET, X, Y, "ConvSIFTNET", is_sift=True)

    # Test combined model
    print("Testing Combined Model...")
    start_time = time()
    combined_predictions = (predictions_v1 + predictions_v2 + predictions_sift) / 3.0
    end_time = time()
    combined_classes = np.argmax(combined_predictions, axis=1)
    true_classes = np.argmax(Y, axis=1)
    combined_acc = accuracy_score(true_classes, combined_classes)
    combined_time = end_time - start_time
    model_results.append(("Combined Model", combined_acc * 100, combined_time))

    # Print results
    print("\n==== Model Performance ====")
    for model_name, acc, time_taken in model_results:
        print(f"Model: {model_name}")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Time taken: {time_taken:.2f} seconds\n")

    # Print confusion matrix and classification report for Combined Model
    print(f"Confusion Matrix for Combined Model:\n{confusion_matrix(true_classes, combined_classes)}")
    print(f"Classification Report for Combined Model:\n{classification_report(true_classes, combined_classes)}")

# Load test data
print("Loading test data...")
X = np.load("dataset/Fer2013_X_test.npy")
Y = np.load("dataset/Fer2013_Y_test.npy")
Test_Accuracy_And_Speed(X, Y)
