# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import numpy as np
import pickle
import fnmatch
import features
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog
import cv2
from tensorflow.keras.models import model_from_json # type: ignore
import time



# Loading model into UI
# CNN_1
json_model = open("models/CNN_v1/CNN_v1_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_CNN_1 = model_from_json(loaded_json_model)
model_CNN_1.load_weights("models/CNN_v1/CNN_v1_Fer2013_best_weights.keras")

# CNN_2
json_model = open("models/CNN_v2/CNN_v2_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_CNN_2 = model_from_json(loaded_json_model)
model_CNN_2.load_weights("models/CNN_v2/CNN_v2_Fer2013_final_weights.keras")

# model_CNN_2.summary()

# SIFTNET
json_model = open("models/CNN_SIFT/ConvSIFTNET_Fer2013_model.json", 'r')
loaded_json_model = json_model.read()
json_model.close()
model_SIFTNET = model_from_json(loaded_json_model)
model_SIFTNET.load_weights("models/CNN_SIFT/ConvSIFTNET_Fer2013_final_model.keras")

# Kmean model to extract SIFT features
Kmean_SIFT = pickle.load(open("models/descriptors/SIFT_Detector_Kmean_model_1.sav", 'rb'))

emotion_dict = {0: "ANGRY", 1: "DISGUST", 2: "FEAR", 3: "HAPPY", 4: "SAD", 5: "SURPRISE", 6: "NEUTRAL"}

emotion_colors = {
    "ANGRY": (97, 105, 255),       # Red
    "DISGUST": (119, 221, 119),     # Green
    "FEAR": (225, 177, 195),        # Purple
    "HAPPY": (150, 253, 253),     # Yellow
    "SAD": (207, 198, 174),       # Cyan
    "SURPRISE": (152, 200, 250),  # Magenta
    "NEUTRAL": (255, 255, 255)  # White
}

# Face Detection
modelFile = "models/pretrained_opencv/opencv_face_detector_uint8.pb"
configFile = "models/pretrained_opencv/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile) # load the model into the network

# Detecting Emotion in Image
def imageDetect(img):

    start_time = time.time()

    if (img.shape[0] > 720 or img.shape[1] > 1080):
        if(img.shape[0] > img.shape[1]):
            scale_by_H = 0
        else:
            scale_by_H = 1

        if scale_by_H == 0:
            scale_factor = int(img.shape[0] /  720)
        else:
            scale_factor = int(img.shape[1] / 1080)
        dim = None

        if scale_factor > 1:
            scale_percent = (scale_factor-1) * 10
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
        else:
            scale_percent = 100
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
        img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert img to grayscale
    (h, w) = img.shape[:2] # get the height and width of the image

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300),interpolation=cv2.INTER_AREA), 1.0, (300, 300),(104.0,
                                                                                                            177.0,
                                                                                                            123.0)) # create a blob from the image
    # mean rgb value is always 104, 177, 123
    # Truyền dữ liệu vào mạng nhận diện khuôn mặt
    net.setInput(blob) # set the input for the network
    detections = net.forward() # forward pass the image through the network
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # get the bounding box of the face
            (x, y, ex, ey) = box.astype("int")
            width = ex - x
            height = ey - y

            roi_gray = gray[y:ey, x:ex]
            cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)


            sift_bow_vector = features.Vetorize_of_An_Image(cropped_img, Kmean_SIFT)
            sift_bow_vector = [sift_bow_vector]
            sift_bow_vector = np.array(sift_bow_vector)

            cropped_img = np.reshape(cropped_img, (-1, 48, 48, 1))
            cropped_img = cropped_img / 255.0 # normalize the image

            predicted_V1 = model_CNN_1.predict(cropped_img)
            predicted_V2 = model_CNN_2.predict(cropped_img)
            predicted_SIFT = model_SIFTNET.predict([cropped_img, sift_bow_vector])

            predicted_combine = (predicted_SIFT + predicted_V1 + predicted_V2) / 3.0 # predicted_combine's type is numpy array
            top = predicted_combine[0].argsort()[-2:][::-1] # get the top 2 highest ratings
            print(predicted_combine[0])
            print(predicted_combine[0].argsort())
            print(top)

            emotion = emotion_dict[top[0]]
            color = emotion_colors[emotion]

            Prob1 = predicted_combine[0][top[0]]
            Prob2 = predicted_combine[0][top[1]]
            print(Prob1)
            print(Prob2)

            fontScale = (width + height) / (width * height) + 0.4

            cv2.rectangle(img, (x - 20, y), (ex + 20, ey), color , 1, cv2.LINE_AA)
            cv2.rectangle(img, (x - 20, ey), (ex + 20, ey + 20), color , -1)
            cv2.putText(img, f"{emotion_dict[top[0]]}: {Prob1:.2f}", (x - 15, ey + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), 1, cv2.LINE_AA)

            if Prob2 > 0.05:
                cv2.rectangle(img, (x - 20, ey + 20), (ex + 20, ey + 20 + 15), color, -1)
                cv2.putText(img, f"{emotion_dict[top[1]]}: {Prob2:.2f}", (x - 15, ey + 15 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale,(0, 0, 0), 1, cv2.LINE_AA)

    end_time = time.time()
    print(f"Time taken for emotion recognition: {end_time - start_time} seconds")
    cv2.imshow('Result', img)


# Detecting Emotion in Video
def videoDetect(path):
    cap = cv2.VideoCapture(path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    cap.set(cv2.CAP_PROP_FPS, 60.0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cap.get(3)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(cap.get(4)))
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q') or cap.isOpened()== 0:
            cap.release()
            cv2.destroyAllWindows()
            return
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) #
        # preprocess the image and prepare it for classification
        net.setInput(blob) # set the input for the network so that it can be classified
        detections = net.forward() # forward pass the image through the network to get the detections
        (h, w) = frame.shape[:2]
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, ex, ey) = box.astype("int")
                width = x + ex
                height = y + ey

                roi_gray = gray[y:ey, x:ex]
                cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                sift_bow_vector = features.Vetorize_of_An_Image(cropped_img, Kmean_SIFT)
                sift_bow_vector = [sift_bow_vector]
                sift_bow_vector = np.array(sift_bow_vector)

                cropped_img = np.reshape(cropped_img, (-1, 48, 48, 1))
                cropped_img = cropped_img / 255.0

                predicted_V1 = model_CNN_1.predict(cropped_img)
                predicted_V2 = model_CNN_2.predict(cropped_img)
                predicted_SIFT = model_SIFTNET.predict([cropped_img, sift_bow_vector])
                predicted_combine = (predicted_SIFT + predicted_V1 + predicted_V2) / 3.0

                top = predicted_combine[0].argsort()[-2:][::-1]

                emotion = emotion_dict[top[0]]
                color = emotion_colors[emotion]

                Prob1 = predicted_combine[0][top[0]]
                Prob2 = predicted_combine[0][top[1]]

                fontScale = (width + height) / (width * height) + 0.4

                cv2.rectangle(frame, (x - 20, y), (ex + 20, ey), color, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x - 20, ey), (ex + 20, ey + 20), color, -1)
                cv2.putText(frame, f"{emotion_dict[top[0]]}: {Prob1:.2f}", (x - 15, ey + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                            (0, 0, 0), 1, cv2.LINE_AA)
                if Prob2 > 0.05:
                    cv2.rectangle(frame, (x - 20, ey + 20), (ex + 20, ey + 20 + 15), color, -1)
                    cv2.putText(frame, f"{emotion_dict[top[1]]}: {Prob2:.2f}", (x - 15, ey + 15 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('PRESS Q TO EXIT', frame)


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(389, 600)
        # Form.setFixedSize(389,532)
        Form.setFixedSize(389,600)
        Form.setWindowIcon(QtGui.QIcon('icon.png'))
        Form.setContentsMargins(0, 0, 0, 0)

        # Background Image
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(-190, -100, 751, 631))
        self.label.setStyleSheet("image: url(/Users/hiimbias/PycharmProjects/FED/image/howareyou-animation-v2.gif);")
        self.label.setText("")
        self.label.setObjectName("label")

        # Open Camera Button
        self.btn_opencamera = QtWidgets.QPushButton(Form)
        self.btn_opencamera.setGeometry(QtCore.QRect(120, 480, 151, 25))
        self.btn_opencamera.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_opencamera.setObjectName("btn_opencamera")

        # File Path Label
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(50, 520, 51, 16))
        self.label_2.setObjectName("label_2")

        # Browse Button
        self.btn_browse = QtWidgets.QPushButton(Form)
        self.btn_browse.setGeometry(QtCore.QRect(100, 550, 75, 25))

        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)

        self.btn_browse.setFont(font)
        self.btn_browse.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_browse.setObjectName("btn_browse")

        # File Path Line
        self.filenameLine = QtWidgets.QLineEdit(Form)
        self.filenameLine.setGeometry(QtCore.QRect(110, 520, 185, 20))
        self.filenameLine.setObjectName("filenameLine")

        # Detect Button
        self.btn_detect = QtWidgets.QPushButton(Form)
        self.btn_detect.setGeometry(QtCore.QRect(180, 550, 125, 25))
        self.btn_detect.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_detect.setObjectName("btn_detect")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        # Button Click Events
        self.btn_opencamera.clicked.connect(self.opencamera)
        self.btn_detect.clicked.connect(self.detect)
        self.btn_browse.clicked.connect(self.browse)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "How Are You?"))
        self.btn_opencamera.setText(_translate("Form", "Open Your Camera"))
        self.label_2.setText(_translate("Form", "File Path:"))
        self.btn_browse.setText(_translate("Form", "Browse"))
        self.btn_detect.setText(_translate("Form", "Scan & Detect"))

    # Emotion Detection in Real Time
    def opencamera(self):
        cv2.destroyAllWindows()
        stream = cv2.VideoCapture(0)
        stream.set(cv2.CAP_PROP_FPS, 60.0)
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  # get the version of cv2

        if int(major_ver) < 3:
            fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else:
            fps = stream.get(cv2.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        while True:
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stream.release()
                    cv2.destroyAllWindows()
                    return
                ret, frame = stream.read() # read the frame from the camera
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame to grayscale
                try:
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                except:
                    print("Exception Blob Resize")
                    continue
                net.setInput(blob)
                detections = net.forward()
                (h, w) = frame.shape[:2]

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.5:
                        start_time = time.time()  # Start timing

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, ex, ey) = box.astype("int")
                        width = x + ex
                        height = y + ey

                        roi_gray = gray[y:ey, x:ex]

                        cropped_img = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                        sift_bow_vector = features.Vetorize_of_An_Image(cropped_img, Kmean_SIFT)
                        sift_bow_vector = [sift_bow_vector]
                        sift_bow_vector = np.array(sift_bow_vector)

                        cropped_img = np.reshape(cropped_img, (-1, 48, 48, 1))
                        cropped_img = cropped_img / 255.0

                        predicted_V1 = model_CNN_1.predict(cropped_img)
                        predicted_V2 = model_CNN_2.predict(cropped_img)
                        predicted_SIFT = model_SIFTNET.predict([cropped_img, sift_bow_vector])

                        predicted_combine = (predicted_SIFT + predicted_V1 + predicted_V2) / 3.0

                        top = predicted_combine[0].argsort()[-2:][::-1]

                        emotion = emotion_dict[top[0]]
                        color = emotion_colors[emotion]

                        Prob1 = predicted_combine[0][top[0]]  # pick the highest rating
                        Prob2 = predicted_combine[0][top[1]]  # pick the second highest rating

                        fontScale = (width + height) / (width * height) + 0.4
                        cv2.rectangle(frame, (x - 20, y), (ex + 20, ey), color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x - 20, ey), (ex + 20, ey + 20), color, -1)
                        cv2.putText(frame, f"{emotion_dict[top[0]]}: {Prob1:.2f}", (x - 15, ey + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                                    (0, 0, 0), 1, cv2.LINE_AA)
                        if Prob2 > 0.05:  # if the rating is higher than threshold
                            cv2.rectangle(frame, (x - 20, ey + 20), (ex + 20, ey + 20 + 15), color, -1)
                            cv2.putText(frame, f"{emotion_dict[top[1]]}: {Prob2:.2f}", (x - 15, ey + 15 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 1, cv2.LINE_AA)

                        end_time = time.time()
                        print(f"Time taken for emotion recognition: {end_time - start_time} seconds")

                cv2.imshow('PRESS Q TO EXIT', frame)

            except:
                print("Warning : Bound Box Out of Frame !")
                continue
    # browse file base on file type
    def browse(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Browse", "",
                                                  "Image Files (*.png *.jpg *.tiff  *.jpeg);; Video Files (*.avi *.mp4 *.mov)")
        if fileName:
            print(fileName)
        self.filenameLine.setText(fileName)


    # discriminate between image and video
    def detect(self):

        if fnmatch.fnmatch(self.filenameLine.text(), "*.mp4") or fnmatch.fnmatch(self.filenameLine.text(),
                                                                                 "*.avi") or fnmatch.fnmatch(
                self.filenameLine.text(), "*.mov"):
            videoDetect(self.filenameLine.text())
        else:
            img = cv2.imread(self.filenameLine.text())
            imageDetect(img)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv) # create an application
    Form = QtWidgets.QWidget() # create a widget
    ui = Ui_Form() # create an instance of the class
    ui.setupUi(Form) # set up the UI
    Form.show() # show the UI
    sys.exit(app.exec_()) # execute the application
