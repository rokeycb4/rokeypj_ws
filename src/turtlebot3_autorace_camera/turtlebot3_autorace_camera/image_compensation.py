#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult

class ImageCompensation(Node):

    def __init__(self):
        super().__init__('image_compensation')
        # 파라미터 선언 (대비 보정에 사용할 clip_hist_percent)
        parameter_descriptor_clip_hist = ParameterDescriptor(
            description='clip hist range.',
            floating_point_range=[FloatingPointRange(
                from_value=0.0, 
                to_value=10.0,
                step=0.1)]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                (
                    'camera.extrinsic_camera_calibration.clip_hist_percent',
                    1.0,
                    parameter_descriptor_clip_hist
                ),
                ('is_extrinsic_camera_calibration_mode', False)
            ]
        )
        self.clip_hist_percent = self.get_parameter(
            'camera.extrinsic_camera_calibration.clip_hist_percent'
        ).value
        self.is_calibration_mode = self.get_parameter('is_extrinsic_camera_calibration_mode').value

        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.param_update_callback)
            # 노란색 트랙바 창
            cv2.namedWindow("HSV Trackbars", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Lower Hue", "HSV Trackbars", 7, 179, lambda x: None)
            cv2.createTrackbar("Lower Sat", "HSV Trackbars", 73, 255, lambda x: None)
            cv2.createTrackbar("Lower Val", "HSV Trackbars", 89, 255, lambda x: None)
            cv2.createTrackbar("Upper Hue", "HSV Trackbars", 47, 179, lambda x: None)
            cv2.createTrackbar("Upper Sat", "HSV Trackbars", 233, 255, lambda x: None)
            cv2.createTrackbar("Upper Val", "HSV Trackbars", 249, 255, lambda x: None)
            
            # 흰색 트랙바 창 추가
            cv2.namedWindow("White Mask Trackbars", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Lower White H", "White Mask Trackbars", 0, 179, lambda x: None)
            cv2.createTrackbar("Lower White S", "White Mask Trackbars", 0, 255, lambda x: None)
            cv2.createTrackbar("Lower White V", "White Mask Trackbars", 200, 255, lambda x: None)
            cv2.createTrackbar("Upper White H", "White Mask Trackbars", 179, 179, lambda x: None)
            cv2.createTrackbar("Upper White S", "White Mask Trackbars", 70, 255, lambda x: None)
            cv2.createTrackbar("Upper White V", "White Mask Trackbars", 255, 255, lambda x: None)
            
            # compensation 값을 조절하는 트랙바 창 (1~10의 값, 실제 clip_hist_percent는 *2하여 2~20이 적용)
            cv2.namedWindow("Compensation Trackbar", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Clip Percent", "Compensation Trackbar", 1, 10, lambda x: None)

        # 구독/발행 메시지 타입: compressed 사용
        self.sub_image_type = 'compressed'
        self.pub_image_type = 'compressed'

        # 원본 이미지 구독
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage,
                '/camera/preprocessed/compressed',
                self.cbImageCompensation,
                10
            )
        else:
            self.sub_image_original = self.create_subscription(
                Image,
                '/image_raw',
                self.cbImageCompensation,
                10
            )

        # 보정된 이미지 발행
        if self.pub_image_type == 'compressed':
            self.pub_image_compensated = self.create_publisher(
                CompressedImage,
                '/image_compensated/compressed',
                10
            )
        else:
            self.pub_image_compensated = self.create_publisher(
                Image,
                '/image_compensated',
                10
            )

        # 추가: 흰색, 노란색 마스크 이미지 발행 publisher 생성
        if self.pub_image_type == 'compressed':
            self.pub_white_mask = self.create_publisher(
                CompressedImage,
                '/image_compensated/white_mask/compressed',
                10
            )
            self.pub_yellow_mask = self.create_publisher(
                CompressedImage,
                '/image_compensated/yellow_mask/compressed',
                10
            )
        else:
            self.pub_white_mask = self.create_publisher(
                Image,
                '/image_compensated/white_mask',
                10
            )
            self.pub_yellow_mask = self.create_publisher(
                Image,
                '/image_compensated/yellow_mask',
                10
            )

        self.cvBridge = CvBridge()

    def param_update_callback(self, parameters):
        for param in parameters:
            self.get_logger().info(f'Parameter name: {param.name}')
            self.get_logger().info(f'Parameter value: {param.value}')
            self.get_logger().info(f'Parameter type: {param.type_}')
            if param.name == 'camera.extrinsic_camera_calibration.clip_hist_percent':
                self.clip_hist_percent = param.value
        self.get_logger().info(f'change: {self.clip_hist_percent}')
        return SetParametersResult(successful=True)
    
    def keep_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)
        largest = max(contours, key=cv2.contourArea)
        result = np.zeros_like(mask)
        cv2.drawContours(result, [largest], -1, 255, thickness=cv2.FILLED)
        return result

    def cbImageCompensation(self, msg_img):
        # 이미지 디코딩
        if self.sub_image_type == 'compressed':
            np_image_original = np.frombuffer(msg_img.data, np.uint8)
            cv_image_original = cv2.imdecode(np_image_original, cv2.IMREAD_COLOR)
        else:
            cv_image_original = self.cvBridge.imgmsg_to_cv2(msg_img, 'bgr8')

        # compensation 값 선택: calibration mode이면 트랙바 값을 사용 (1~10 값을 2배하여 2~20)
        if self.is_calibration_mode:
            clip_value = cv2.getTrackbarPos("Clip Percent", "Compensation Trackbar")
            clip_hist_percent = float(clip_value * 2)
        else:
            clip_hist_percent = self.clip_hist_percent

        # 대비 보정 진행 (그레이스케일 히스토그램 기반)
        cv_image_compensated = np.copy(cv_image_original)
        hist_size = 256
        gray = cv2.cvtColor(cv_image_compensated, cv2.COLOR_BGR2GRAY)
        if clip_hist_percent == 0.0:
            min_gray, max_gray, _, _ = cv2.minMaxLoc(gray)
        else:
            hist = cv2.calcHist([gray], [0], None, [hist_size], [0, hist_size])
            accumulator = np.cumsum(hist)
            total_max = accumulator[hist_size - 1]
            clip_hist_percent_adjusted = clip_hist_percent * (total_max / 100.0) / 2.0
            min_gray = 0
            while accumulator[min_gray] < clip_hist_percent_adjusted and min_gray < hist_size - 1:
                min_gray += 1
            max_gray = hist_size - 1
            while accumulator[max_gray] >= (total_max - clip_hist_percent_adjusted) and max_gray > 0:
                max_gray -= 1
        input_range = max_gray - min_gray
        alpha = (hist_size - 1) / input_range if input_range != 0 else 1.0
        beta = -min_gray * alpha
        self.get_logger().info(
            f"[Clip {clip_hist_percent}] min_gray: {min_gray}, max_gray: {max_gray}, alpha: {alpha:.3f}, beta: {beta:.3f}"
        )
        cv_image_compensated = cv2.convertScaleAbs(cv_image_compensated, alpha=alpha, beta=beta)

        # 보정된 이미지 발행
        if self.pub_image_type == 'compressed':
            msg_comp = self.cvBridge.cv2_to_compressed_imgmsg(cv_image_compensated, 'jpg')
        else:
            msg_comp = self.cvBridge.cv2_to_imgmsg(cv_image_compensated, 'bgr8')

        msg_comp.header = msg_img.header 
        self.pub_image_compensated.publish(msg_comp)
        # HSV 변환 및 마스크 적용
        hsv_img = cv2.cvtColor(cv_image_compensated, cv2.COLOR_BGR2HSV)

        # 흰색 마스크: calibration mode이면 trackbar 값을 사용, 아니면 기본값
        if self.is_calibration_mode:
            lower_white = np.array([
                cv2.getTrackbarPos("Lower White H", "White Mask Trackbars"),
                cv2.getTrackbarPos("Lower White S", "White Mask Trackbars"),
                cv2.getTrackbarPos("Lower White V", "White Mask Trackbars")
            ])
            upper_white = np.array([
                cv2.getTrackbarPos("Upper White H", "White Mask Trackbars"),
                cv2.getTrackbarPos("Upper White S", "White Mask Trackbars"),
                cv2.getTrackbarPos("Upper White V", "White Mask Trackbars")
            ])
        else:
            # lower_white = np.array([44, 0, 249])
            # upper_white = np.array([120, 255, 255])
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([179, 70, 255])
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)

        # 노란색 마스크: calibration mode이면 HSV trackbar 값을 사용, 아니면 기본값
        if self.is_calibration_mode:
            lower_yellow = np.array([
                cv2.getTrackbarPos("Lower Hue", "HSV Trackbars"),
                cv2.getTrackbarPos("Lower Sat", "HSV Trackbars"),
                cv2.getTrackbarPos("Lower Val", "HSV Trackbars")
            ])
            upper_yellow = np.array([
                cv2.getTrackbarPos("Upper Hue", "HSV Trackbars"),
                cv2.getTrackbarPos("Upper Sat", "HSV Trackbars"),
                cv2.getTrackbarPos("Upper Val", "HSV Trackbars")
            ])
        else:
            # lower_yellow = np.array([15, 0, 89])
            # upper_yellow = np.array([47, 101, 255])
            lower_yellow = np.array([2, 0, 32])
            upper_yellow = np.array([52, 255, 255])
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        # ✅ 노이즈 제거: 가장 큰 덩어리만 남기기
        white_mask = self.keep_largest_contour(white_mask)
        yellow_mask = self.keep_largest_contour(yellow_mask)

        if self.is_calibration_mode:
            cv2.imshow("Yellow Mask Calibration", yellow_mask)
            cv2.imshow("White Mask Calibration", white_mask)
            cv2.waitKey(1)

        # 마스크 결과 발행(1채널) #jpg의 경우 에러 발생(JPEG라는 이미지 형식의 '태생'과 '목적' 때문) 따라서 png(이미지 형식을 명확하게 저장함)로 수정
        ## 1채널 마스크(white_mask)를 'jpg'(jpeg)로 바로 압축하면, 받는 쪽(detect_lane 노드)에서 채널 수 문제로 CvBridgeError가 발생
        if self.pub_image_type == 'compressed':
            msg_white_mask = self.cvBridge.cv2_to_compressed_imgmsg(white_mask, 'png')
            msg_yellow_mask = self.cvBridge.cv2_to_compressed_imgmsg(yellow_mask, 'png')

        # mono8[흑백], bgr8(순수 이미지)은 압축방식으로 사용x
        else:
            msg_white_mask = self.cvBridge.cv2_to_imgmsg(white_mask, 'mono8')
            msg_yellow_mask = self.cvBridge.cv2_to_imgmsg(yellow_mask, 'mono8')

    # [추가할 코드] 원본 이미지의 헤더를 그대로 복사
        msg_white_mask.header = msg_img.header
        msg_yellow_mask.header = msg_img.header

        self.pub_white_mask.publish(msg_white_mask)
        self.pub_yellow_mask.publish(msg_yellow_mask)

def main(args=None):
    rclpy.init(args=args)
    node = ImageCompensation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


