import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

# def Vocabularyize(X_filename, K=128):
#     X = np.load(X_filename, allow_pickle=True)
#     detector = cv2.SIFT_create()
#     all_descriptors = []
#     for img in X:
#         img = img.astype(np.uint8)
#         keypoints, descriptors = detector.detectAndCompute(img, None)
#         if descriptors is not None:
#             all_descriptors.extend(descriptors)
#
#     if len(all_descriptors) == 0:
#         print("No descriptors found. Exiting.")
#         return
#
#     all_descriptors = np.array(all_descriptors)
#     print(f"Total descriptors collected: {all_descriptors.shape}")
#
#     print("Clustering descriptors...")
#     K_model = MiniBatchKMeans(n_clusters=K, max_iter=300, batch_size=K * 2, max_no_improvement=30, init_size=3 * K)
#     K_model.fit(all_descriptors)
#     print("KMeans clustering completed.")
#
#     filename = f"SIFT_Detector_Kmean_model_1.sav"
#     pickle.dump(K_model, open(filename, 'wb'))
#     print("KMeans model saved.")



def Vocabularyize(X_filename,K=2048):

    X = np.load(X_filename)
    print("Clustering !!!!")
    K_model = MiniBatchKMeans(n_clusters=K,max_iter=300,batch_size=K*2,max_no_improvement=30,init_size=3*K).fit(X)
    print("Clustered !!!!")
    # save the model to disk
    filename = "generator/Fer2013_Detector_Kmean_model.sav"
    pickle.dump(K_model, open(filename, 'wb'))
    print("Model Saved !!!!")

# Tệp chứa descriptor đã được tạo ở bước 1
descriptor_file = "/generator/NEW_SIFTDescriptors_FER2013.npy"

# Số lượng cluster để tạo từ vựng
# num_clusters = 100

# Tạo từ vựng thị giác
Vocabularyize(
    X_filename=descriptor_file
)