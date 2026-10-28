import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class CatImageClassifier:
    """
    定义一个用于调用 Google Teachable Machine 的模型来对猫的种类进行分类
    """
    def __init__(self):
        # 初始化 TensorFlow Lite 解释器，加载 Google Teachable Machine 模型
        self.model = tf.lite.Interpreter(model_path="GoogleTMModel/model.tflite")
        # 分配张量内存
        self.model.allocate_tensors()
        # 获取模型的输入和输出细节
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        # 打开标签文件，读取所有标签并存储在列表中
        with open("GoogleTMModel/labels.txt", "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def predict_image(self, image_path):
        # 打开指定路径的图像，并将其转换为 RGB 格式
        image = Image.open(image_path).convert("RGB")
        # 将图像调整为模型所需的输入尺寸 (224 * 224)
        image = image.resize((224, 224))
        # 将图像转换为 numpy 数组
        image_array = np.asarray(image)
        # 对图像数组进行归一化处理，将像素值缩放到 [-1, 1] 范围
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        # 在数组的第一个维度上添加一个维度，以匹配模型的输入形状
        normalized_image_array = np.expand_dims(normalized_image_array, axis=0)

        # 将归一化后的图像数组设置为模型的输入
        self.model.set_tensor(self.input_details[0]['index'], normalized_image_array)
        # 调用模型进行推理
        self.model.invoke()
        # 获取模型的输出预测结果
        prediction = self.model.get_tensor(self.output_details[0]['index'])

        # 找到所有标签中最长的标签长度，用于格式化输出
        max_class_name_length = max(len(self.labels[i].strip()) for i in range(len(self.labels)))
        # 遍历预测结果中的每个类别得分，并且格式化打印标签和分数
        for i, score in enumerate(prediction[0]):
            class_name = self.labels[i].strip()
            formatted_class_name = class_name.ljust(max_class_name_length)
            print(f"Class: {formatted_class_name} Confidence Score: {score * 100:.2f}%")

    def real_time_prediction(self):
        # 打开默认摄像头（设备 ID 为 0）
        cap = cv2.VideoCapture(0)

        # 定义进度条的颜色列表
        bar_colors = [
            (1, 119, 230),
            (111, 76, 216),
            (239, 74, 121),
            (210, 103, 25)
        ]

        # 初始化空白区域（用于显示进度条）和计数器（用于控制推理频率）
        blank_space = None
        count = 0
        while True:
            # 从摄像头读取一帧图像
            ret, frame = cap.read()
            # 如果读取失败，跳出循环
            if not ret:
                break

            # 每 3 帧进行一次推理
            # 每一帧推理一次并且在屏幕上更新一次分类概率太快了，眼睛看不清
            # 所以通过以下逻辑进行降速
            if count % 3 == 0:
                count = 1
                # 将 OpenCV 读取的 BGR 格式图像转换为 RGB 格式，并转换为 PIL 图像
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 以下对于图像的预处理以及模型推理过程都与图像静态预测相同
                image = image.resize((224, 224))
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                normalized_image_array = np.expand_dims(normalized_image_array, axis=0)

                self.model.set_tensor(self.input_details[0]['index'], normalized_image_array)
                self.model.invoke()
                prediction = self.model.get_tensor(self.output_details[0]['index'])

                # 通过构造两个圆角矩形，并且求得两个圆角矩形的交集，用于标识进度条
                # 定义进度条宽度
                progress_bar_width = 300
                # 创建一个空白的 numpy 数组，用于显示进度条
                blank_space = np.zeros((frame.shape[0], progress_bar_width, 3), dtype=np.uint8)
                # 计算进度条的内边距
                padding = int(progress_bar_width / 8)
                # 计算进度条开始的垂直位置
                start_y = frame.shape[0] - (len(prediction[0]) * 30 + 20)

                # 遍历预测结果中的每个类别得分
                for i, score in enumerate(prediction[0]):
                    class_name = self.labels[i].strip()
                    temp1_start = padding
                    temp1_end = progress_bar_width - padding
                    temp2_start = padding + int(score * (progress_bar_width - 2 * padding)) - progress_bar_width
                    temp2_end = padding + int(score * (progress_bar_width - 2 * padding))
                    bar_y_top = start_y + i * 30
                    bar_y_bottom = start_y + 20 + i * 30
                    bar_height = bar_y_bottom - bar_y_top
                    radius = bar_height // 4

                    color = bar_colors[i % len(bar_colors)]

                    # temp1 定义第一个圆角矩形
                    temp1 = np.zeros_like(blank_space)
                    # temp1 绘制中间矩形部分
                    cv2.rectangle(temp1, (temp1_start + radius, bar_y_top), (temp1_end - radius, bar_y_bottom), color, -1)
                    # temp1 绘制左侧
                    cv2.ellipse(temp1, (temp1_start + radius, bar_y_top + radius), (radius, radius), 90, 90, 180, color, -1)
                    cv2.rectangle(temp1, (temp1_start, bar_y_top + radius), (temp1_start + radius, bar_y_bottom - radius), color, -1)
                    cv2.ellipse(temp1, (temp1_start + radius, bar_y_bottom - radius), (radius, radius), 90, 0, 90, color, -1)
                    # temp1 绘制右侧
                    cv2.ellipse(temp1, (temp1_end - radius, bar_y_top + radius), (radius, radius), 90, 180, 270, color, -1)
                    cv2.rectangle(temp1, (temp1_end - radius, bar_y_top + radius), (temp1_end, bar_y_bottom - radius), color, -1)
                    cv2.ellipse(temp1, (temp1_end - radius, bar_y_bottom - radius), (radius, radius), 90, 270, 360, color, -1)

                    # temp2 定义第二个圆角矩形
                    temp2 = np.zeros_like(blank_space)
                    # temp2 绘制中间矩形部分
                    cv2.rectangle(temp2, (temp2_start + radius, bar_y_top), (temp2_end - radius, bar_y_bottom), color, -1)
                    # temp1 绘制左侧
                    cv2.ellipse(temp2, (temp2_start + radius, bar_y_top + radius), (radius, radius), 90, 90, 180, color, -1)
                    cv2.rectangle(temp2, (temp2_start, bar_y_top + radius), (temp2_start + radius, bar_y_bottom - radius), color, -1)
                    cv2.ellipse(temp2, (temp2_start + radius, bar_y_bottom - radius), (radius, radius), 90, 0, 90, color, -1)
                    # 绘制右侧
                    cv2.ellipse(temp2, (temp2_end - radius, bar_y_top + radius), (radius, radius), 90, 180, 270, color, -1)
                    cv2.rectangle(temp2, (temp2_end - radius, bar_y_top + radius), (temp2_end, bar_y_bottom - radius), color, -1)
                    cv2.ellipse(temp2, (temp2_end - radius, bar_y_bottom - radius), (radius, radius), 90, 270, 360, color, -1)

                    # 取两个临时图像的交集
                    intersection = cv2.bitwise_and(temp1, temp2)

                    # 将交集结果合并到空白区域
                    blank_space = cv2.bitwise_or(blank_space, intersection)

                    # 将进度条和分类概率都打印到屏幕上去
                    cv2.putText(blank_space, f"{class_name}: {score * 100:.2f}%", (temp1_start + 10, start_y + 15 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # 没有到达 3 帧时，直接复制上一帧的空白区域
                count = count + 1
                blank_space = blank_space

            window_name = 'Real-time Cat Breeds Prediction'
            frame = np.hstack((frame, blank_space))
            cv2.imshow(window_name, frame)

            if ((cv2.waitKey(1) & 0xFF) == ord('q')) or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
                break

        cap.release()
        cv2.destroyAllWindows()
