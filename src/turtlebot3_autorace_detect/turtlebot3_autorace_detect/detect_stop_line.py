import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class StopLineDetector(Node):
    def __init__(self):
        super().__init__('detect_stopline')
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )
        self.pub_stopline = self.create_publisher(Bool, '/detect/stopline', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # 압축 이미지 → OpenCV 이미지로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ROI: 이미지 하단 120픽셀 (720x1280 기준)
        roi = frame[600:720, :]

        # Grayscale 변환 + 이진화
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 전체 흰색 픽셀 수 계산
        white_pixel_count = cv2.countNonZero(binary)

        # 윤곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 유효한 수직 띠 개수 세기
        valid_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 20 and w > 5 and h > w:
                valid_contours += 1

        # 조건 1: 띠가 3개 이상
        enough_stripes = valid_contours >= 3

        # 조건 2: 흰색 픽셀 수가 충분한가
        enough_white = white_pixel_count > 1000

        # 두 조건 모두 만족해야 stop
        stopline_detected = enough_stripes and enough_white

        # 결과 발행
        msg = Bool()
        msg.data = stopline_detected
        self.pub_stopline.publish(msg)

        # 디버깅 로그 출력
        self.get_logger().info(
            f'Stopline detected: {stopline_detected} | White pixel count: {white_pixel_count} | Valid contours: {valid_contours}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = StopLineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
