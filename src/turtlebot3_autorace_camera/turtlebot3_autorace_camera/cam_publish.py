#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2

class LaptopCamPublisher(Node):
    def __init__(self):
        super().__init__('laptop_cam_publisher')

        # 퍼블리시할 토픽 (autorace 시스템 호환성을 위해 그대로 유지)
        self.publisher_ = self.create_publisher(
            CompressedImage, '/camera/image_raw/compressed', 10)

        # 노트북 기본 내장카메라 열기 (보통 0번 장치)
        self.cap = cv2.VideoCapture(0)

        # 카메라 파라미터 설정 (노트북 카메라 해상도는 일부 장치에서 제한 가능)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f"Laptop camera resolution: {width} x {height}")

        if not self.cap.isOpened():
            self.get_logger().error("노트북 카메라 열기 실패!")
            exit(1)
        else:
            self.get_logger().info("노트북 카메라 오픈 성공")

        # 0.2초 (약 5fps) 주기로 퍼블리시
        self.timer = self.create_timer(0.2, self.publish_image)

    def publish_image(self):
        ret, frame = self.cap.read()

        if ret:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            _, compressed_image = cv2.imencode('.jpg', frame, encode_param)

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "laptop_camera"
            msg.format = "jpeg"
            msg.data = compressed_image.tobytes()

            self.publisher_.publish(msg)
            self.get_logger().info('Published laptop /camera/image_raw/compressed')
        else:
            self.get_logger().warning('노트북 카메라 프레임 읽기 실패')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaptopCamPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
