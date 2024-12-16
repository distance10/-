import cv2
import numpy as np
import os
import re
import base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
class FaceRecognizer:
    def __init__(self, model_dir, names):
        self.model_dir = model_dir
        self.names = names
        self.face_cascade = cv2.CascadeClassifier("D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("start")
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

    def recognize_face(self, img):
        """识别传入的图像中的人脸"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("start detect")
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        results = []
        print("faces:", faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for model_file in os.listdir(model_dir):
                # 初始化识别器
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(r'./Model/' + model_file)
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                name = "unknown"
                print("confidence:", confidence)
                print(idnum)
                if confidence < 120 and idnum in self.models:
                    # 计算相似度
                    similarity = 1 / (1 + confidence)  # 计算相似度
                    similarity_percentage = similarity * 100  # 转换为百分比
                    print("similarity percentage:", similarity_percentage)
                    name = self.names[idnum - 1]
                    results.append({"name": name, "similarity": similarity_percentage})

        return results, img

# 在 Flask 应用中定义 FaceRecognizer 实例
face_recognizer = None

@app.route('/')
def index():
    """返回前端页面"""
    return render_template('face_test.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """接收前端上传的图像并进行人脸识别"""
    global face_recognizer  # 使用全局变量
    print(face_recognizer)
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # 打印接收到的数据
        print(f"Received data: {data}")

        # 获取图像数据并解码
        image_data = data['image'].split(',')[1]  # 去掉前缀部分
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

        # 检查图像是否成功解码
        if img is None:
            return jsonify({"error": "Image decoding failed"}), 400

        print("yes")
        # 进行人脸识别
        if face_recognizer is None:
            return jsonify({"error": "Face recognizer not initialized"}), 500

        print("yes2")
        results, img_with_faces = face_recognizer.recognize_face(img)

        # 将处理后的图像保存或返回
        output_path = os.path.join('./uploads', 'output_image.jpg')
        cv2.imwrite(output_path, img_with_faces)

        # 确保返回的结果是一个数组
        return jsonify({"results": results}), 200

    except Exception as e:
        print(f"Error processing image: {e}")  # 打印错误信息
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    # 初始化 FaceRecognizer 实例
    model_dir = './Model/'
    names = ['吴晨锋', '马浩然', '韦汉雨', '杨金鹏']
    face_recognizer = FaceRecognizer(model_dir, names)
    print("start")

    # 启动 Flask 应用
    app.run(debug=True)