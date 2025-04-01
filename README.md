**I. Cấu trúc Dự án** 

1. **Thư mục chính**

Thư mục FED/: Thư mục gốc của dự án, chứa toàn bộ mã nguồn, dữ liệu của dự án.	

2. **Thư mục con và nội dung.**  
1. data/:   
* Chứa các tập dữ liệu sử dụng cho huấn luyện và kiểm thử.  
* File Fer\_X.npy và Fer\_Y.npy lần lượt chứa dữ liệu đặc trưng được biểu diễn bằng giá trị các điểm ảnh và nhãn cảm xúc tương ứng được gắn với hình ảnh đó.  
2. models/ :   
* CNN\_v1/ : Chứa mô hình CNN\_v1.  
* CNN\_v2/ : Chứa mô hình CNN\_v2.  
* CNN\_SIFT/ : Chứa mô hình CNN\_SIFT.  
* pretrained\_opencv/ : Chứa mô hình huấn luyện sẵn để phát hiện gương mặt của OpenCV.   
* descriptors: chứa những file chứa dữ liệu sau: dữ liệu trích xuất từ hình ảnh, dữ liệu bao gồm các từ điển trực quan được khởi tạo sử dụng thuật toán K-means, dữ liệu bao gồm các vectơ chứa thông tin về tần suất “từ trực quan” ứng với đặc trưng trong hình ảnh.  
3. sift\_bovw/ : Chứa các file để thực hiện quá trình trích xuất các đặc trưng sử dụng phương pháp SIFT và BOVW.  
4. features.py: Mã nguồn chứa những phương thức sử dụng để trích xuất đặc trưng SIFT và BOVW.  
5. confusion\_rsrc: Chứa những file sử dụng cho quá trình đánh giá mô hình.  
6. speed\_test.py: File chứa mã nguồn sử dụng để kiểm tra, đánh giá mô hình.  
7. cnn.py: File chứa mã nguồn để huấn luyện mô hình.  
8. ConfusionMatrix.py: File chứa mã nguồn để trực quan hoá ma trận nhầm lẫn.  
9. UI.py: FIle chứa mã nguồn của giao diện trực quan mô hình.  
10. requirements.txt: Danh sách thư viện cần thiết (Tensorflow, OpenCV, NumPy, v.v.).

**II. Tài liệu Hướng dẫn Sử dụng**  
Giao diện tương tác mô hình tổ hợp Nhận diện cảm xúc 

1. **Giới thiệu**

Giao diện tương tác mô hình tổ hợp Nhận diện cảm xúc thực hiện 3 tác vụ phân tích chính:  
Phân tích biểu cảm khuôn mặt từ hình ảnh.  
Phân tích biểu cảm khuôn mặt từ video.  
Phân tích biểu cảm khuôn mặt theo thời gian thực.  
Kết quả đầu ra bao gồm nhận diện các cảm xúc: Tức giận, Chán ghét \- Ghê tởm, Sợ hãi, Hạnh phúc, Buồn bã, Ngạc nhiên, Bình thường \- Trung tính.

2. **Yêu cầu hệ thống**

Hệ điều hành: WIndows, macOS, Linux  
Phần mềm: Python 3.8+, OpenCV, TensorFlow/Keras, PyQt5

3. **Cài đặt**

3.1. Tải về mã nguồn  
Tải mã nguồn từ link sau

**3.2. Cài đặt môi trường**   
Cài đặt Python và các thư viện yêu cầu

| pip install \-r requirements.txt |
| :---- |

Tải mô hình đã huấn luyện tại đây

**3.3. Chạy ứng dụng**  
Khởi chạy giao diện tương tác

| python path-to-UI.py-file |
| :---- |

**III. Hướng dẫn sử dụng giao diện tương tác**

1. **Khởi chạy giao diện tương tác sử dụng lệnh** 

Cửa sổ giao diện tương tác xuất hiện 

2. **Tính năng Nhận diện cảm xúc trong hình ảnh.**  
* Để sử dụng tính năng này đầu tiên người dùng phải đưa hình ảnh cần được nhận diện vào trong ứng dụng thông qua nút Browse.  
* Một cửa sổ trình quản lý file sẽ hiện lên cho phép người dùng lựa chọn file ảnh mong muốn đưa vào hệ thống.  
* Đảm bảo lựa chọn Image Files để lựa chọn hình ảnh.  
* Lựa chọn hình ảnh và nhấn Open để đưa hình ảnh muốn nhận diện vào mô hình.  
* Sau đó, nhấn Scan & Detect để thực hiện tác vụ Nhận diện cảm xúc trong hình ảnh.  
* Một cửa sổ hiện lên với hình ảnh vừa được đưa vào đi kèm với kết quả Nhận diện cảm xúc trong bức ảnh

3. **Tính năng nhận diện cảm xúc qua video**  
* Tương tự như tính năng nhận diện cảm xúc qua ảnh, ở đây người dùng cần lựa chọn Video Files để đưa video muốn nhận diện vào trong ứng dụng.  
* Lựa chọn video mong muốn và nhấn Open để đưa video vào trong ứng dụng.  
* Ứng dụng sẽ trả về video và thực hiện Nhận dạng cảm xúc trong video theo thời gian thực.

4. **Tính năng nhận diện cảm xúc theo thời gian thực thông qua camera**  
* Để sử dụng tính năng này, người dùng cần lựa chọn vào nút Open Your Camera để mở camera của thiết bị lên.  
* Sau khi lựa chọn nút Open Your Camera, một cửa sổ sẽ hiện lên và trình chiếu những hình ảnh được thu từ camera kèm với kết quả.

