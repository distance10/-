import os
import cv2
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
# 获取分类器
detector = cv2.CascadeClassifier("D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml")

# 获取图像及标签
def get_images_and_labels(image_directory):
    image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]
    face_samples = []
    labels = []
    print('正在训练人脸...')
    for image_path in image_paths:
        label = 3
        img_array = load_and_convert_image(image_path)
        detected_faces = detector.detectMultiScale(img_array)

        for (x, y, w, h) in detected_faces:
            face_samples.append(img_array[y:y + h, x: x + w])
            labels.append(label)

    return face_samples, labels


# 加载并转换图像
def load_and_convert_image(image_path):
    pil_image = Image.open(image_path).convert('L')
    return np.array(pil_image, 'uint8')

# 主程序入口
if __name__ == "__main__":
    path = 'Face_img/3/'
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))
    print('训练已完成！')
    recognizer.write(r'./Model/model_3.yml')