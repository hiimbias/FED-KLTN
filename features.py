import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import pickle

def Extractor(Images):
    Detector = cv2.SIFT_create()
    Descriptor = cv2.SIFT_create()

    desc_seq = []
    count = 0
    for img in Images:
        if img is None or img.size == 0:
            print(f"Image Number {count} is empty or corrupted, skipping.")
            count += 1
            continue

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp = Detector.detect(img)
        kp, desc = Descriptor.compute(img, kp)

        if desc is None or len(kp) == 0:
            print(f"No keypoints found in Image Number {count}, skipping.")
            count += 1
            continue

        print(f"Image Number {count} has been extracted!")
        desc_seq.append(desc)
        count += 1

    print("Images Extracted! Concatenating descriptors...")
    descriptors_data = np.concatenate(desc_seq, axis=0) if desc_seq else np.empty((0, 128))
    filename = f"NEW_SIFTDescriptors_FER2013.npy"
    np.save(filename, descriptors_data)
    print(f"{filename} has been saved to disk!")

def Vocabularyize(X_filename, data_name_code, K=128, detector_name="FAST"):
    X = np.load(X_filename, allow_pickle=True)
    detector = cv2.SIFT_create() if detector_name == "SIFT" else cv2.FastFeatureDetector_create()

    all_descriptors = []
    for img in X:
        img = img.astype(np.uint8)
        keypoints, descriptors = detector.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.extend(descriptors)

    all_descriptors = np.array(all_descriptors)
    print(f"Total descriptors collected: {all_descriptors.shape}")

    print("Clustering descriptors...")
    K_model = MiniBatchKMeans(n_clusters=K, max_iter=300, batch_size=K * 2, max_no_improvement=30, init_size=3 * K)
    K_model.fit(all_descriptors)
    print("KMeans clustering completed.")

    filename = f"SIFT_Detector_Kmean_model_1.sav"
    pickle.dump(K_model, open(filename, 'wb'))
    print("KMeans model saved.")


def Vetorize_of_An_Image(img, Kmean):
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(img, None)

    # Ensure the descriptors are reshaped to match the expected input shape
    if descriptors is not None:
        descriptors = descriptors.reshape(-1, 128)
        vector_2048 = np.zeros(Kmean.n_clusters, dtype=int)
        predictions = Kmean.predict(descriptors)
        for label in predictions:
            vector_2048[label] += 1
    else:
        vector_2048 = np.zeros(Kmean.n_clusters, dtype=int)

    return vector_2048

def Histogram_All_Images(imgs, Kmean, detector_method="SIFT", data_name_code=1):
    Stack = []
    for count, img in enumerate(imgs):
        vector = Vetorize_of_An_Image(img, Kmean, detector_method)
        Stack.append(vector)
        print(f"Image Number {count} in Stack!")

    Stack = np.array(Stack)
    print("Histogram Generated!")
    filename = f"1_SIFT_Detector_Histogram.npy"
    np.save(filename, Stack)
    print(f"Saved Histogram as numpy array to disk: {filename}")





