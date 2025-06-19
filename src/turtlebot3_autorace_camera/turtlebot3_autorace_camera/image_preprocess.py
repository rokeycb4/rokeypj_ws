#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')

        self.declare_parameter('pp', True)
        self.use_preprocessing = self.get_parameter('pp').get_parameter_value().bool_value
        self.get_logger().info(f'전처리 모드: {self.use_preprocessing}')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            CompressedImage,
            '/camera/preprocessed/compressed',
            10
        )

    def adjust_gamma(self, image, gamma=1.6):
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)

    def reduce_v_channel(self, image, scale=0.8):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v * scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.use_preprocessing:
            self.get_logger().info('전처리 적용됨: CLAHE + 대비감소 + 감마 + V채널 억제')

            # CLAHE (명암 향상)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 대비/밝기 조절
            processed = cv2.convertScaleAbs(processed, alpha=0.9, beta=-30)

            # 감마 보정으로 빛반사 억제
            processed = self.adjust_gamma(processed, gamma=1.6)

            # V 채널 줄이기 (고휘도 영역 약화)
            processed = self.reduce_v_channel(processed, scale=0.85)

        else:
            processed = frame
            self.get_logger().info('전처리 미적용 (원본 그대로)')

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, compressed_image = cv2.imencode('.jpg', processed, encode_param)

        out_msg = CompressedImage()
        out_msg.header = msg.header
        out_msg.format = "jpeg"
        out_msg.data = compressed_image.tobytes()

        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
