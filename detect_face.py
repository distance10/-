import cv2
import numpy as np
import os
import re
class FaceRecognizer:
    def __init__(self, model_dir, names):
        self.model_dir = model_dir
        self.names = names
        self.face_cascade = cv2.CascadeClassifier("D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_models()

    def load_models(self):
        """加载所有模型文件"""
        self.models = {}
        for model_file in os.listdir(self.model_dir):
            if model_file.endswith('.yml'):
                # 提取ID
                idnum = int(re.search(r'\d+', model_file).group(0))
                model_path = os.path.join(self.model_dir, model_file)

                self.recognizer.read(model_path)  # 加载模型
                self.models[idnum] = model_file  # 将模型与ID关联
                print(f"模型 {model_file} 加载成功，ID 关联为 {idnum}")

    def recognize_faces(self):
        """启动摄像头并识别面孔"""
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        minW = 0.1 * camera.get(3)
        minH = 0.1 * camera.get(4)
        font = cv2.FONT_HERSHEY_SIMPLEX

        print('请正对着摄像头...')
        name = "unknown"
        while True:
            success, img = camera.read()
            if not success:
                print("无法读取摄像头图像")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(int(minW), int(minH)))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for model_file in os.listdir(model_dir):

                    # 初始化识别器
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(r'./Model/' + model_file)
                    idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    print(f"idnum: {idnum} + confidence: {confidence}")
                    if confidence < 120 and idnum in self.models:

                        # 计算相似度
                        similarity = 1 / (1 + confidence)  # 计算相似度
                        similarity_percentage = similarity * 100  # 转换为百分比
                        print("相似度:", similarity_percentage)
                        name = self.names[idnum - 1]
                        print(f"ID: {idnum}, NAME: {name}, confidence: {confidence}, similarity: {similarity_percentage}")

                cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (230, 250, 100), 1)
                cv2.putText(img, f"置信度: {round(confidence, 2)}", (x + 5, y + h - 5), font, 1, (255, 0, 0), 1)

            cv2.imshow('camera', img)
            if cv2.waitKey(10) == 27:  # 按Esc键退出
                break

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_dir = './Model/'
    names = ['Wu chenfeng', 'Ma haoran', 'Wei hanyu', 'Yang jinpeng']
    face_recognizer = FaceRecognizer(model_dir, names)
    face_recognizer.recognize_faces()