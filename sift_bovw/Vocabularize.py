import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

def Vocabularyize(X_filename,K=2048):

    X = np.load(X_filename)
    print("Clustering !!!!")
    K_model = MiniBatchKMeans(n_clusters=K,max_iter=300,batch_size=K*2,max_no_improvement=30,init_size=3*K).fit(X)
    print("Clustered !!!!")
    # save the model to disk
    filename = "../generator/Fer2013_Detector_Kmean_model.sav"
    pickle.dump(K_model, open(filename, 'wb'))
    print("Model Saved !!!!")


descriptor_file = "/generator/NEW_SIFTDescriptors_FER2013.npy"

Vocabularyize(
    X_filename=descriptor_file
)