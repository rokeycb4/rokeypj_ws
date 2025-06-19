
import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane_test')

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=179,
                step=1)]
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            description='saturation and lightness range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=255,
                step=1)]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detect.lane.white.hue_l', 0,
                    parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179,
                    parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 0,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.saturation_h', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_l', 105,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.hue_l', 10,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 127,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.saturation_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_l', 95,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('is_detection_calibration_mode', False)
            ]
        )

        self.hue_white_l = self.get_parameter(
            'detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter(
            'detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter(
            'detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter(
            'detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = self.get_parameter(
            'detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = self.get_parameter(
            'detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'detect.lane.yellow.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_type = 'compressed' 
        self.pub_image_type = 'compressed' 

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                #CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                CompressedImage, '/image_raw/compressed', self.cbFindLane, 1                
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1
                )
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1
                )

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1
                    )
            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1
                    )

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)

        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
            )

        self.pub_white_line_reliability = self.create_publisher(
            UInt8, '/detect/white_line_reliability', 1
            )

        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)
        
        # 디버깅용 이미지 퍼블리셔 추가
        if self.pub_image_type == 'compressed':
            self.debug_pub = self.create_publisher(
                CompressedImage, '/detect/lane_debug/compressed', 1
                )
        else:
            self.debug_pub = self.create_publisher(
                Image, '/detect/lane_debug', 1
                )

        self.cvBridge = CvBridge()

        self.counter = 1

        self.window_width = 1000.
        self.window_height = 600.

        self.reliability_white_line = 100
        self.reliability_yellow_line = 100

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))
        
        # 차선 검출 파라미터
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 40  # 낮춰서 더 많은 선 검출
        self.min_line_length = 80  # 짧은 선도 검출
        self.max_line_gap = 100    # 더 큰 갭 허용
        
        # 이동 평균 필터 파라미터
        self.prev_centerx = None
        self.alpha = 0.3  # 이동 평균 가중치 (높을수록 현재 프레임 반영 비중 증가)
        
        # 이전 프레임의 차선 정보 저장
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.lane_width_px = 400  # 예상 차선 폭 (픽셀)
        
        # ROI 마스크 좌표
        self.roi_vertices = np.array([
            [(100, 600), (400, 350), (880, 350), (1180, 600)]
        ], dtype=np.int32)

    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            self.get_logger().info(f'Parameter name: {param.name}')
            self.get_logger().info(f'Parameter value: {param.value}')
            self.get_logger().info(f'Parameter type: {param.type_}')
            if param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.white.hue_h':
                self.hue_white_h = param.value
            elif param.name == 'detect.lane.white.saturation_l':
                self.saturation_white_l = param.value
            elif param.name == 'detect.lane.white.saturation_h':
                self.saturation_white_h = param.value
            elif param.name == 'detect.lane.white.lightness_l':
                self.lightness_white_l = param.value
            elif param.name == 'detect.lane.white.lightness_h':
                self.lightness_white_h = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.hue_h':
                self.hue_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.saturation_l':
                self.saturation_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.saturation_h':
                self.saturation_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.lightness_l':
                self.lightness_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.lightness_h':
                self.lightness_yellow_h = param.value
            return SetParametersResult(successful=True)

    def preprocess_image(self, image):
        """이미지 전처리 함수 - 빛 조건에 더 강인하게 개선"""
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # CLAHE 적용 - 대비 향상으로 빛 조건 변화에 더 강인하게 만듦
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 이진화 - 적응형 임계값 적용으로 조명 변화에 강인하게 함
        binary = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def create_roi_mask(self, image):
        """ROI 마스크 생성"""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image):
        """에지 검출 - 이미 이진화된 이미지에서 에지 검출"""
        # 이미지가 이미 이진화되어 있으므로 Canny 대신 모폴로지 연산 사용
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return edges

    def detect_lines(self, edges):
        """허프 변환을 통한 직선 검출"""
        return cv2.HoughLinesP(edges, 1, np.pi/180, 
                             threshold=self.hough_threshold,
                             minLineLength=self.min_line_length,
                             maxLineGap=self.max_line_gap)

    def separate_lines(self, lines):
        """검출된 선을 좌/우 차선으로 분리"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.3:  # 왼쪽 차선
                left_lines.append(line)
            elif slope > 0.3:  # 오른쪽 차선
                right_lines.append(line)
            
        return left_lines, right_lines

    def fit_lane_lines(self, lines, image_shape):
        """차선 피팅"""
        if not lines:
            return None
        
        points = np.concatenate([line[0] for line in lines])
        x = points[:, [0, 2]].flatten()
        y = points[:, [1, 3]].flatten()
        
        try:
            coefficients = np.polyfit(y, x, deg=1)
            line_fit = np.poly1d(coefficients)
            
            y1 = image_shape[0]
            y2 = int(image_shape[0] * 0.6)
            x1 = int(line_fit(y1))
            x2 = int(line_fit(y2))
            
            return np.array([[x1, y1, x2, y2]])
        except:
            return None

    def calculate_lane_center(self, left_fit, right_fit, image_shape):
        """차선 중심점 계산"""
        if left_fit is None or right_fit is None:
            return None

        left_x = left_fit[0][0]
        right_x = right_fit[0][0]
        center_x = (left_x + right_x) // 2
        
        return center_x

    def cbFindLane(self, image_msg):
        """차선 검출 콜백 함수"""
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # 기존 HSV 기반 차선 검출 방식
        white_fraction, cv_white_lane = self.maskWhiteLane(cv_image)
        yellow_fraction, cv_yellow_lane = self.maskYellowLane(cv_image)
        
        # 새로운 빛에 강인한 차선 검출 방식
        processed = self.preprocess_image(cv_image)
        masked = self.create_roi_mask(processed)
        edges = self.detect_edges(masked)
        lines = self.detect_lines(edges)
        
        # 결과 이미지 준비
        result_image = cv_image.copy()
        
        # 기존 차선 검출 로직
        try:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.fit_from_lines(
                    self.left_fit, cv_yellow_lane)
                self.mov_avg_left = np.append(
                    self.mov_avg_left, np.array([self.left_fit]), axis=0
                    )

            if white_fraction > 3000:
                self.right_fitx, self.right_fit = self.fit_from_lines(
                    self.right_fit, cv_white_lane)
                self.mov_avg_right = np.append(
                    self.mov_avg_right, np.array([self.right_fit]), axis=0
                    )
        except Exception:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.sliding_windown(cv_yellow_lane, 'left')
                self.mov_avg_left = np.array([self.left_fit])

            if white_fraction > 3000:
                self.right_fitx, self.right_fit = self.sliding_windown(cv_white_lane, 'right')
                self.mov_avg_right = np.array([self.right_fit])

        MOV_AVG_LENGTH = 5

        self.left_fit = np.array([
            np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])
        self.right_fit = np.array([
            np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])

        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]

        if self.mov_avg_right.shape[0] > 1000:
            self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]
        
        # 새로운 차선 검출 방식으로 보완
        if lines is not None:
            # 좌/우 차선 분리 및 피팅
            left_lines, right_lines = self.separate_lines(lines)
            left_fit_new = self.fit_lane_lines(left_lines, cv_image.shape)
            right_fit_new = self.fit_lane_lines(right_lines, cv_image.shape)
            
            # 결과 이미지에 새로운 방식으로 검출된 차선 표시
            if left_fit_new is not None:
                cv2.line(result_image, 
                        (left_fit_new[0][0], left_fit_new[0][1]), 
                        (left_fit_new[0][2], left_fit_new[0][3]), 
                        (255, 0, 0), 3)
                
            if right_fit_new is not None:
                cv2.line(result_image, 
                        (right_fit_new[0][0], right_fit_new[0][1]), 
                        (right_fit_new[0][2], right_fit_new[0][3]), 
                        (0, 255, 0), 3)
            
            # 기존 차선 검출이 실패했을 경우 새로운 방식으로 보완
            if yellow_fraction <= 3000 and left_fit_new is not None:
                self.left_fitx, self.left_fit = self.sliding_windown(cv_yellow_lane, 'left')
                self.mov_avg_left = np.array([self.left_fit])
                
            if white_fraction <= 3000 and right_fit_new is not None:
                self.right_fitx, self.right_fit = self.sliding_windown(cv_white_lane, 'right')
                self.mov_avg_right = np.array([self.right_fit])
                
            # 양쪽 차선이 모두 검출되지 않았을 경우 이전 프레임의 정보 활용
            if yellow_fraction <= 3000 and self.prev_left_fit is not None:
                self.left_fit = self.prev_left_fit
                
            if white_fraction <= 3000 and self.prev_right_fit is not None:
                self.right_fit = self.prev_right_fit
                
            # 현재 검출된 차선 정보 저장
            if yellow_fraction > 3000:
                self.prev_left_fit = self.left_fit
            if white_fraction > 3000:
                self.prev_right_fit = self.right_fit

        # 디버깅용 결과 이미지 발행
        if self.pub_image_type == 'compressed':
            self.debug_pub.publish(self.cvBridge.cv2_to_compressed_imgmsg(result_image, 'jpg'))
        else:
            self.debug_pub.publish(self.cvBridge.cv2_to_imgmsg(result_image, 'bgr8'))

        # 기존 make_lane 호출
        self.make_lane(cv_image, white_fraction, yellow_fraction)

    def maskWhiteLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_white_l
        Hue_h = self.hue_white_h
        Saturation_l = self.saturation_white_l
        Saturation_h = self.saturation_white_h
        Lightness_l = self.lightness_white_l
        Lightness_h = self.lightness_white_h

        lower_white = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_white = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        fraction_num = np.count_nonzero(mask)

        if not self.is_calibration_mode:
            if fraction_num > 35000:
                if self.lightness_white_l < 250:
                    self.lightness_white_l += 5
            elif fraction_num < 5000:
                if self.lightness_white_l > 50:
                    self.lightness_white_l -= 5

        how_much_short = 0

        for i in range(0, 600):
            if np.count_nonzero(mask[i, ::]) > 0:
                how_much_short += 1

        how_much_short = 600 - how_much_short

        if how_much_short > 100:
            if self.reliability_white_line >= 5:
                self.reliability_white_line -= 5
        elif how_much_short <= 100:
            if self.reliability_white_line <= 99:
                self.reliability_white_line += 5

        msg_white_line_reliability = UInt8()
        msg_white_line_reliability.data = self.reliability_white_line
        self.pub_white_line_reliability.publish(msg_white_line_reliability)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
                    )

            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
                    )

        return fraction_num, mask

    def maskYellowLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_yellow_l
        Hue_h = self.hue_yellow_h
        Saturation_l = self.saturation_yellow_l
        Saturation_h = self.saturation_yellow_h
        Lightness_l = self.lightness_yellow_l
        Lightness_h = self.lightness_yellow_h

        lower_yellow = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_yellow = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        fraction_num = np.count_nonzero(mask)

        if self.is_calibration_mode:
            if fraction_num > 35000:
                if self.lightness_yellow_l < 250:
                    self.lightness_yellow_l += 20
            elif fraction_num < 5000:
                if self.lightness_yellow_l > 90:
                    self.lightness_yellow_l -= 20

        how_much_short = 0

        for i in range(0, 600):
            if np.count_nonzero(mask[i, ::]) > 0:
                how_much_short += 1

        how_much_short = 600 - how_much_short

        if how_much_short > 100:
            if self.reliability_yellow_line >= 5:
                self.reliability_yellow_line -= 5
        elif how_much_short <= 100:
            if self.reliability_yellow_line <= 99:
                self.reliability_yellow_line += 5

        msg_yellow_line_reliability = UInt8()
        msg_yellow_line_reliability.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_yellow_lane.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
                    )

            elif self.pub_image_type == 'raw':
                self.pub_image_yellow_lane.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
                    )

        return fraction_num, mask

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_inds = (
            (nonzerox >
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
            (nonzerox <
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin))
                )

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        lane_fit = np.polyfit(y, x, 2)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def sliding_windown(self, img_w, left_or_right):
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)

        out_img = np.dstack((img_w, img_w, img_w)) * 255

        midpoint = np.int_(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        elif left_or_right == 'right':
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20

        window_height = np.int_(img_w.shape[0] / nwindows)

        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_current = lane_base

        margin = 50

        minpix = 50

        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            cv2.rectangle(
                out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            good_lane_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &
                (nonzerox < win_x_high)
                ).nonzero()[0]

            lane_inds.append(good_lane_inds)

            if len(good_lane_inds) > minpix:
                x_current = np.int_(np.mean(nonzerox[good_lane_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except Exception:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def make_lane(self, cv_image, white_fraction, yellow_fraction):
        # Create an image to draw the lines on
        warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)

        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

        # both lane -> 2, left lane -> 1, right lane -> 3, none -> 0
        lane_state = UInt8()

        if yellow_fraction > 3000:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_left]),
                isClosed=False,
                color=(0, 0, 255),
                thickness=25
                )

        if white_fraction > 3000:
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_right]),
                isClosed=False,
                color=(255, 255, 0),
                thickness=25
                )

        self.is_center_x_exist = True

        if self.reliability_white_line > 50 and self.reliability_yellow_line > 50:
            if white_fraction > 3000 and yellow_fraction > 3000:
                centerx = np.mean([self.left_fitx, self.right_fitx], axis=0)
                pts = np.hstack((pts_left, pts_right))
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 2

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

                # Draw the lane onto the warped blank image
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            if white_fraction > 3000 and yellow_fraction <= 3000:
                centerx = np.subtract(self.right_fitx, 280)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 3

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

            if white_fraction <= 3000 and yellow_fraction > 3000:
                centerx = np.add(self.left_fitx, 280)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 1

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

        elif self.reliability_white_line <= 50 and self.reliability_yellow_line > 50:
            centerx = np.add(self.left_fitx, 280)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 1

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        elif self.reliability_white_line > 50 and self.reliability_yellow_line <= 50:
            centerx = np.subtract(self.right_fitx, 280)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 3

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        else:
            self.is_center_x_exist = False

            lane_state.data = 0

            pass

        self.pub_lane_state.publish(lane_state)
        self.get_logger().info(f'Lane state: {lane_state.data}')

        # Combine the result with the original image
        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

        if self.pub_image_type == 'compressed':
            if self.is_center_x_exist:
                # publishes lane center
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(350)
                
                # 이동 평균 필터로 중심점 안정화
                if self.prev_centerx is None:
                    self.prev_centerx = centerx.item(350)
                else:
                    # 급격한 변화 감지 및 제한
                    max_change = 50  # 최대 허용 변화량
                    diff = centerx.item(350) - self.prev_centerx
                    if abs(diff) > max_change:
                        centerx_filtered = self.prev_centerx + (max_change if diff > 0 else -max_change)
                    else:
                        centerx_filtered = centerx.item(350)
                    
                    # 이동 평균 적용
                    centerx_filtered = self.alpha * centerx_filtered + (1 - self.alpha) * self.prev_centerx
                    self.prev_centerx = centerx_filtered
                    msg_desired_center.data = centerx_filtered
                    
                self.pub_lane.publish(msg_desired_center)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

        elif self.pub_image_type == 'raw':
            if self.is_center_x_exist:
                # publishes lane center
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(350)
                
                # 이동 평균 필터로 중심점 안정화
                if self.prev_centerx is None:
                    self.prev_centerx = centerx.item(350)
                else:
                    # 급격한 변화 감지 및 제한
                    max_change = 50  # 최대 허용 변화량
                    diff = centerx.item(350) - self.prev_centerx
                    if abs(diff) > max_change:
                        centerx_filtered = self.prev_centerx + (max_change if diff > 0 else -max_change)
                    else:
                        centerx_filtered = centerx.item(350)
                    
                    # 이동 평균 적용
                    centerx_filtered = self.alpha * centerx_filtered + (1 - self.alpha) * self.prev_centerx
                    self.prev_centerx = centerx_filtered
                    msg_desired_center.data = centerx_filtered
                    
                self.pub_lane.publish(msg_desired_center)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(final, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
