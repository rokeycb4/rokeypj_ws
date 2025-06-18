#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class BEVCalibrator(Node):
    def __init__(self):
        super().__init__('bev_calibrator')
        self.cvBridge = CvBridge()

        # 보정된 이미지를 구독
        self.create_subscription(
            CompressedImage,
            '/image_compensated/compressed',
            self.image_callback,
            10)
        
        # 이미지 크기 설정
        self.img_width = 1280
        self.img_height = 720

        # 트랙바를 위한 윈도우 생성
        cv2.namedWindow("BEV Controls")
        
        # 4개 꼭짓점의 x, y 좌표를 위한 8개의 트랙바 생성
        # 초기값은 이전에 사용했던 좌표값으로 설정
        cv2.createTrackbar("Top-Left x", "BEV Controls", 200, self.img_width, self.on_trackbar)
        cv2.createTrackbar("Top-Left y", "BEV Controls", 400, self.img_height, self.on_trackbar)
        
        cv2.createTrackbar("Bottom-Left x", "BEV Controls", 50, self.img_width, self.on_trackbar)
        cv2.createTrackbar("Bottom-Left y", "BEV Controls", 480, self.img_height, self.on_trackbar)

        cv2.createTrackbar("Bottom-Right x", "BEV Controls", 450, self.img_width, self.on_trackbar)
        cv2.createTrackbar("Bottom-Right y", "BEV Controls", 480, self.img_height, self.on_trackbar)
        
        cv2.createTrackbar("Top-Right x", "BEV Controls", 380, self.img_width, self.on_trackbar)
        cv2.createTrackbar("Top-Right y", "BEV Controls", 400, self.img_height, self.on_trackbar)
        
        self.get_logger().info("BEV Calibrator node has been initialized.")
        self.get_logger().info("Adjust trackbars and press 's' in the terminal to print coordinates.")

    def on_trackbar(self, val):
        # 트랙바가 변경될 때마다 호출되지만, 실제 처리는 image_callback에서 하므로 내용은 비워둡니다.
        pass

    def image_callback(self, msg):
        # 1. 이미지 디코딩
        image = self.cvBridge.compressed_imgmsg_to_cv2(msg)

        # 2. 트랙바에서 현재 좌표값들을 읽어오기
        tl_x = cv2.getTrackbarPos("Top-Left x", "BEV Controls")
        tl_y = cv2.getTrackbarPos("Top-Left y", "BEV Controls")
        bl_x = cv2.getTrackbarPos("Bottom-Left x", "BEV Controls")
        bl_y = cv2.getTrackbarPos("Bottom-Left y", "BEV Controls")
        br_x = cv2.getTrackbarPos("Bottom-Right x", "BEV Controls")
        br_y = cv2.getTrackbarPos("Bottom-Right y", "BEV Controls")
        tr_x = cv2.getTrackbarPos("Top-Right x", "BEV Controls")
        tr_y = cv2.getTrackbarPos("Top-Right y", "BEV Controls")

        # 3. 원근 변환을 위한 소스(src) 및 목적지(dst) 좌표 정의
        src = np.float32([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
        dst = np.float32([[0, 0], [0, self.img_height], [self.img_width, self.img_height], [self.img_width, 0]])

        # 4. 원근 변환 매트릭스 계산 및 BEV 이미지 생성
        M = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(image, M, (self.img_width, self.img_height), flags=cv2.INTER_LINEAR)

        # 5. 시각화: 원본 이미지에 ROI(관심 영역) 그리기
        roi_image = image.copy()
        # cv2.polylines는 점들을 순서대로 연결합니다.
        cv2.polylines(roi_image, [np.int32(src)], isClosed=True, color=(0, 255, 0), thickness=2)

        # 6. 창에 이미지 표시
        cv2.imshow("Original Image with ROI", roi_image)
        cv2.imshow("BEV Result", warped_img)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            self.get_logger().info("="*20)
            self.get_logger().info("BEV Coordinates Saved!")
            self.get_logger().info(f"src = np.float32([\n    ({tl_x}, {tl_y}),\n    ({bl_x}, {bl_y}),\n    ({br_x}, {br_y}),\n    ({tr_x}, {tr_y})\n])")
            self.get_logger().info("="*20)


def main(args=None):
    rclpy.init(args=args)
    node = BEVCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()