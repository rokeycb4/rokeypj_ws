#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')

        # 구독: CompressedImage 입력 받음
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        # 발행: 전처리 후 다시 CompressedImage 로 발행 (detect_lane2 호환)
        self.publisher_ = self.create_publisher(
            CompressedImage,
            '/camera/preprocessed/compressed',
            10
        )

        self.get_logger().info('전처리 노드 시작 (ROI 제거 & CompressedImage 발행)')

    def image_callback(self, msg):
        # 압축 해제 → OpenCV 이미지
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 색공간 변환 (BGR → LAB)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE (대비 향상)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 블러 적용 (노이즈 완화)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 샤프닝 (경계 강조)
        kernel_sharpening = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)

        # 다시 JPEG 압축 후 CompressedImage로 발행
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, compressed_image = cv2.imencode('.jpg', sharpened, encode_param)

        out_msg = CompressedImage()
        out_msg.header = msg.header  # 원본 헤더 유지
        out_msg.format = "jpeg"
        out_msg.data = compressed_image.tobytes()

        self.publisher_.publish(out_msg)
        self.get_logger().info('전처리 이미지 발행 (CompressedImage 타입)')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
