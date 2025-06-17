#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')

        # Declare parameter pp (preprocessing option)
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

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.use_preprocessing:
            # LAB 변환 → CLAHE 대비보정 → 블러 → 샤프닝
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            kernel_sharpening = np.array([[-1, -1, -1],
                                           [-1, 9, -1],
                                           [-1, -1, -1]])
            processed = cv2.filter2D(blurred, -1, kernel_sharpening)
            self.get_logger().info('전처리 적용됨')
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
