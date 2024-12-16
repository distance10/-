import cv2
import numpy as np

def pad_and_resize(image, size=(64, 64)):
    h, w = image.shape[:2]
    longest = max(h, w)
    top = (longest - h) // 2
    bottom = longest - h - top
    left = (longest - w) // 2
    right = longest - w - left
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded_image, size)

def adjust_brightness(image, alpha=1, bias=0):
    image = np.clip(image.astype(float) * alpha + bias, 0, 255).astype(np.uint8)
    return image

def capture_faces(name, face_id):
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_detector = cv2.CascadeClassifier("D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml")
    count = 1

    while count <= 100:
        print(f"第{face_id} 张人脸，正在采集第 {count}.")
        success, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.31, 2)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = pad_and_resize(face)
            cv2.imwrite(f"Face_img/3/{name}_{face_id}_{count}.jpg", face)
            cv2.putText(img, name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

        cv2.imshow('img', img)
        if cv2.waitKey(30) & 0xff == 27:
            break

    camera.release()#关闭
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('请输入姓名: ')
    face_id = input('请输入人脸编号: ')
    capture_faces(name, face_id)