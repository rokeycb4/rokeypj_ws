#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import threading
import time
import os
import numpy as np

class ImageSaverFromCompressedTopic(Node):
    def __init__(self):
        super().__init__('image_saver_from_compressed_topic')

        self.frame = None
        self.running = True

        # 현재 파일 기준 폴더 생성
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, "saved_img")
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"이미지 저장 폴더: {self.save_dir}")

        # # CompressedImage 토픽 구독
        # self.subscription = self.create_subscription(
        #     CompressedImage,
        #     '/camera/image_raw/compressed',
        #     self.image_callback,
        #     10
        # )

        # CompressedImage 토픽 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/preprocessed/compressed',
            self.image_callback,
            10
        )

        # self.get_logger().info("구독 시작: /camera/image_raw/compressed")
        self.get_logger().info("/camera/preprocessed/compressed")

        # 키보드 입력 감지 스레드 시작
        self.input_thread = threading.Thread(target=self.wait_for_enter)
        self.input_thread.start()

    def image_callback(self, msg):
        # ROS2 CompressedImage → OpenCV 이미지로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.frame = frame

    def wait_for_enter(self):
        while self.running:
            input("Enter 입력 시 수신된 이미지를 저장합니다: ")
            if self.frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.save_dir, f"saved_image_{timestamp}.jpg")
                cv2.imwrite(filename, self.frame)
                self.get_logger().info(f"이미지 저장 완료: {filename}")
            else:
                self.get_logger().warning("아직 수신된 이미지가 없음. 저장 실패.")

    def destroy_node(self):
        self.running = False
        self.input_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverFromCompressedTopic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
