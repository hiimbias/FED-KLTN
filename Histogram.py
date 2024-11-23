import numpy as np
import cv2
import pickle


def Vetorize_of_An_Image(img, Kmean):
    detector = cv2.SIFT_create()

    # Ensure the image is in the correct format
    if img is not None and img.size > 0:
        img = img.astype(np.uint8)
        keypoints, descriptors = detector.detectAndCompute(img, None)
    else:
        keypoints, descriptors = None, None

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


def Histogram_All_Images(imgs, Kmean):
    Stack = Vetorize_of_An_Image(imgs[0], Kmean)
    count = 0
    print("Image Number " + str(count) + " in Stack !")
    for img in imgs[1:]:
        vector = Vetorize_of_An_Image(img, Kmean)
        Stack = np.vstack((Stack, vector))
        count += 1
        print("Image Number " + str(count) + " in Stack !")
    print("Histogram Generated ! ")
    filename = "generator/Fer2013_SIFT_Detector_Histogram.npy"
    np.save(filename, Stack)
    print("Saved Histogram as numpy array to disk  !")


# Load images and KMeans model
image_file = '/dataset/Fer_X.npy'
vocabulary_file = '/generator/Fer2013_Detector_Kmean_model.sav'

imgs = np.load(image_file, allow_pickle=True)
with open(vocabulary_file, 'rb') as f:
    Kmean = pickle.load(f)

# Generate histogram
Histogram_All_Images(imgs, Kmean)