#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2

class CameraImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_image_publisher')

        # 토픽 이름: 반드시 런치파일과 연결되는 이름으로!
        self.publisher_ = self.create_publisher(
            CompressedImage, '/camera/image_raw/compressed', 10)

        # 카메라 디바이스 번호 (노트북: 0, 외부캠: 1)
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.get_logger().error("카메라 열기 실패!")
            exit(1)
        else:
            self.get_logger().info("카메라 오픈 성공")

        # 0.2초마다 이미지 퍼블리시 (약 5fps)
        self.timer = self.create_timer(0.2, self.publish_image)

    def publish_image(self):
        ret, frame = self.cap.read()

        if ret:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            _, compressed_image = cv2.imencode('.jpg', frame, encode_param)

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = compressed_image.tobytes()

            self.publisher_.publish(msg)
            self.get_logger().info('Published /camera/image_raw/compressed')
        else:
            self.get_logger().warning('카메라 프레임 읽기 실패')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
