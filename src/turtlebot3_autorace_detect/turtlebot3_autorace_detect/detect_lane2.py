import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Int32
import cv2
import numpy as np
from cv_bridge import CvBridge


def detect_white_line(cv_image):
    try:
        if cv_image is None or cv_image.size == 0:
            print("입력 이미지가 유효하지 않습니다")
            return None

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        h, w = cv_image.shape[:2]
        mask_roi = np.zeros_like(white_mask)
        mask_roi[int(h * 0.7):, int(w * 0.4):] = 1  # 오른쪽 60%, 하단 30%
        white_mask = white_mask * mask_roi

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return None

        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        if abs(vy) < 0.001:
            return None

        y_bottom = h - 1
        x_bottom = int(((y_bottom - y) * vx / vy) + x)
        y_middle = int(h * 0.25)
        x_middle = int(((y_middle - y) * vx / vy) + x)

        if not (0 <= x_bottom < w and 0 <= x_middle < w):
            return None

        return (x_bottom, y_bottom, x_middle, y_middle)

    except Exception as e:
        print(f"흰색 차선 감지 오류: {e}")
        return None


def detect_yellow_line(cv_image):
    try:
        if cv_image is None or cv_image.size == 0:
            print("입력 이미지가 유효하지 않습니다")
            return None

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 60, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        h, w = cv_image.shape[:2]
        mask_roi = np.zeros_like(yellow_mask)
        mask_roi[int(h * 0.7):, :int(w * 0.6)] = 1  # 왼쪽 60%, 하단 30%
        yellow_mask = yellow_mask * mask_roi

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            return None

        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        if abs(vy) < 0.001:
            return None

        y_bottom = h - 1
        x_bottom = int(((y_bottom - y) * vx / vy) + x)
        y_middle = int(h * 0.25)
        x_middle = int(((y_middle - y) * vx / vy) + x)

        if not (0 <= x_bottom < w and 0 <= x_middle < w):
            return None

        return (x_bottom, y_bottom, x_middle, y_middle)

    except Exception as e:
        print(f"노란색 차선 감지 오류: {e}")
        return None


class WhiteLineTracker(Node):
    def __init__(self):
        super().__init__('white_line_tracker')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/preprocessed/compressed',
            self.image_callback,
            10)

        self.cvBridge = CvBridge()
        self.sub_image_type = 'compressed'

        self.lane_pub = self.create_publisher(Float32, '/lane_center', 10)
        self.lane_state_pub = self.create_publisher(Int32, '/detect/lane_state', 10)
        self.pub_image_lane = self.create_publisher(Image, '/detect/white_image_output', 1)

        self.get_logger().info("흰색 차선 추적 노드가 시작되었습니다.")

    def image_callback(self, msg):
        try:
            if self.sub_image_type == 'compressed':
                np_arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                img = self.cvBridge.imgmsg_to_cv2(msg, 'bgr8')

            if img is None:
                self.get_logger().warn("이미지를 디코딩할 수 없습니다.")
                return

            h, w = img.shape[:2]
            center_x = w // 2

            white_line = detect_white_line(img)

            if white_line:
                x_bottom = white_line[0]
                desired_center = float(x_bottom - 350)  # ← 오프셋 적용
                state = 2
            else:
                desired_center = float(center_x)
                state = 0

            self.lane_pub.publish(Float32(data=desired_center))
            self.lane_state_pub.publish(Int32(data=state))

            # 시각화
            if white_line:
                x_middle, y_middle, x_bottom, y_bottom = white_line
                cv2.rectangle(img, (x_middle, y_middle), (x_bottom, y_bottom), (255, 255, 255), 2)

            yellow_line = detect_yellow_line(img)
            if yellow_line:
                x_middle, y_middle, x_bottom, y_bottom = yellow_line
                cv2.rectangle(img, (x_middle, y_middle), (x_bottom, y_bottom), (255, 255, 0), 2)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(img, 'bgr8'))

            cv2.imshow("Camera View", img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 중 오류 발생: {e}")


def main():
    rclpy.init()
    node = WhiteLineTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
