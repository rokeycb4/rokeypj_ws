import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import time
import sys

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        # 이미지 퍼블리셔 생성
        self.publisher_ = self.create_publisher(CompressedImage, 'image_raw/compressed', 10)
        # 주기적인 이미지 전송을 위한 타이머 설정 (0.2초 주기)
        self.timer = self.create_timer(0.2, self.publish_image)

        self.cap = None
        self.selected_device = None
        self.find_working_camera()

        # 사용 가능한 카메라가 없으면 로그 출력 후 종료 플래그를 설정
        if self.cap is None:
            self.get_logger().error("Available camera not found!")
            self.no_camera = True
        else:
            self.no_camera = False
            self.get_logger().info(f"Using camera device: {self.selected_device}")
            # 카메라 설정 적용
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FPS, 25)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.get_logger().info(
                f"Camera resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
            )

    def find_working_camera(self):
        max_index = 10
        for i in range(max_index):
            device_path = f"/dev/video{i}"
            self.get_logger().info(f"Trying camera device: {device_path}")
            cap_temp = cv2.VideoCapture(device_path)
            # 잠깐 대기 (카메라 초기화를 위한 시간)
            time.sleep(2)
            if cap_temp.isOpened():
                ret, frame = cap_temp.read()
                if ret:
                    self.cap = cap_temp
                    self.selected_device = device_path
                    return
                else:
                    cap_temp.release()
            else:
                cap_temp.release()

    def publish_image(self):
        # 카메라가 없으면 타이머 콜백에서 아무것도 하지 않음.
        if self.no_camera or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            success, compressed_image = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                self.get_logger().error("Failed to encode image")
                return

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = compressed_image.tobytes()

            self.publisher_.publish(msg)
            self.get_logger().info("Publishing compressed image...")
        else:
            self.get_logger().warn("Failed to capture frame")

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()

    # 카메라가 없는 경우, 노드를 종료하고 rclpy shutdown을 호출
    if image_publisher.no_camera:
        image_publisher.get_logger().error("No available camera. Exiting.")
        image_publisher.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        image_publisher.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        if image_publisher.cap is not None:
            image_publisher.cap.release()
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()