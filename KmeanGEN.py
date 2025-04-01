import numpy as np
import pickle
from features import Vocabularyize


X_filename = "/Users/hiimbias/PycharmProjects/FED/dataset/Fer_X.npy"
Y_data = np.load("/Users/hiimbias/PycharmProjects/FED/dataset/Fer_Y.npy")

# Đảm bảo rằng dữ liệu X có định dạng đúng
X_data = np.load(X_filename)
if X_data.ndim == 2:
    X_data = X_data.reshape(-1, 48, 48)  # Giả sử mỗi ảnh là 48x48 pixels
X_data = X_data.astype(np.uint8)

# Tạo mô hình KMeans và lưu mô hình sử dụng để trích xuất dặc trưng thông qua SIFT
data_name_code = 1  # Ví dụ về mã phân biệt dữ liệu
K = 100  # Số cụm KMeans
detector_name = "SIFT"

# Tạo mô hình KMeans và lưu vào file .sav
Vocabularyize(X_filename, data_name_code, K=K, detector_name=detector_name)