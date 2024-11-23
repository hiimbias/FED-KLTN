import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

# Cấu hình tên file và các biến khác
num_clusters = 100  # Số cụm cho KMeans

def Extractor(Images):
    Detector = cv2.SIFT_create()
    Descriptor = cv2.SIFT_create()

    desc_seq = [] # Danh sách chứa các descriptor
    for img in Images:
        # Đảm bảo ảnh có định dạng grayscale và kiểu uint8
        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Nếu là ảnh màu, chuyển sang grayscale
        img = img.astype(np.uint8)  # Đảm bảo kiểu dữ liệu là uint8

        kp = Detector.detect(img) # Tìm keypoint
        kp, desc = Descriptor.compute(img, kp) # Tính toán descriptor
        if desc is not None:
            desc_seq.append(desc)

    # Chuyển đổi các đặc trưng SIFT thành numpy array
    descriptors_data = np.vstack(desc_seq)

    # Bước KMeans clustering để tạo visual words
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=1000, random_state=0)
    kmeans.fit(descriptors_data)

    # Tạo histogram cho mỗi ảnh
    histograms = []
    for desc in desc_seq:
        if desc is not None:
            words = kmeans.predict(desc)
            histogram, _ = np.histogram(words, bins=num_clusters, range=(0, num_clusters))
        else:
            histogram = np.zeros(num_clusters)
        histograms.append(histogram)

    # Chuyển đổi danh sách histogram thành numpy array và lưu file
    histograms_array = np.array(histograms)
    filename = "Fer2013_SIFTDetector_Histogram_GEN.npy"
    np.save(filename, histograms_array)
    print(f"File {filename} đã được tạo.")


# Tải dữ liệu hình ảnh từ file Fer_x.npy
Images = np.load('/Users/hiimbias/PycharmProjects/FED/dataset/Fer_X.npy')

# Gọi hàm Extractor để trích xuất và lưu histogram
Extractor(Images)