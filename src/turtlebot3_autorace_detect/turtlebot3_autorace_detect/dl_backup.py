# ###################################################################### 중가
# 
# ###################################################################### 중간

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8

class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.sub_image_type = 'compressed'
        self.pub_image_type = 'compressed'

        # self.sub_image_original = self.create_subscription(
        #     CompressedImage, '/camera/preprocessed/compressed',
        #     self.cbFindLane, 1
        # )

        self.sub_image_original = self.create_subscription(
            CompressedImage, '/image_compensated/compressed',
            self.cbFindLane, 1
        )

        self.pub_image_lane = self.create_publisher(
            CompressedImage, '/detect/image_output/compressed', 1
        )
        self.pub_image_mask = self.create_publisher(
            CompressedImage, '/detect/yellow_mask/compressed', 1
        )
        self.pub_lane = self.create_publisher(Float64, '/lane_center', 1)
        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
        )
        self.pub_lane_state = self.create_publisher(
            UInt8, '/detect/lane_state', 1
        )
        self.pub_bev_image = self.create_publisher(
            CompressedImage, '/detect/bev_image/compressed', 1
        )

        self.cvBridge = CvBridge()
        self.counter = 1

        self.reliability_yellow_line = 100
        self.mov_avg_left = np.empty((0, 3))

        self.yellow_offset = 300

        self.hsv_yellow_lower = [15, 50, 120]
        self.hsv_yellow_upper = [45, 255, 255]


        # 정확한 원근 변환 행렬 (detect_lane.py 기준)
        src = np.float32([
            [180, 400],  # 왼쪽 위
            [70, 720],   # 왼쪽 아래
            [1230, 720], # 오른쪽 아래
            [1140, 400]  # 오른쪽 위
        ])
        dst = np.float32([
            [0, 0],
            [0, 720],
            [1280, 720],
            [1280, 0]
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)

        self.left_fit = np.array([0.0, 0.0, 300.0])
        self.lane_fit_bef = np.array([0., 0., 0.])

    def cbFindLane(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        np_arr = np.frombuffer(image_msg.data, np.uint8)
        raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        bev_image = cv2.warpPerspective(raw_image, self.M, (1280, 720))

        # BEV 확인용 이미지 발행
        bev_msg = self.cvBridge.cv2_to_compressed_imgmsg(bev_image, 'jpg')
        self.pub_bev_image.publish(bev_msg)

        ploty = np.linspace(0, bev_image.shape[0] - 1, bev_image.shape[0])
        yellow_fraction, mask = self.maskYellowLane(bev_image)

        try:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, mask)
                self.mov_avg_left = np.append(self.mov_avg_left, np.array([self.left_fit]), axis=0)
        except Exception:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.sliding_windown(mask, 'left')
                self.mov_avg_left = np.array([self.left_fit])

        MOV_AVG_LENGTH = 5
        if self.mov_avg_left.shape[0] > 0:
            self.left_fit = np.array([
                np.mean(self.mov_avg_left[::-1][:, 0][:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 1][:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 2][:MOV_AVG_LENGTH])
            ])
        self.left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]

        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]

        self.make_lane(bev_image, yellow_fraction)

        # 노란색 마스크 이미지 발행
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.pub_image_type == 'compressed':
            self.pub_image_mask.publish(self.cvBridge.cv2_to_compressed_imgmsg(mask_bgr, 'jpg'))

    def maskYellowLane(self, image):
        height, width = image.shape[:2]
        roi = image[:, :int(width * 0.6)]

        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.hsv_yellow_lower), np.array(self.hsv_yellow_upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        pad_width = width - mask.shape[1]
        mask = cv2.copyMakeBorder(mask, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

        fraction_num = np.count_nonzero(mask)
        row_nonzero = np.count_nonzero(mask, axis=1)
        self.reliability_yellow_line += 5 if np.count_nonzero(row_nonzero) > 500 else -5
        self.reliability_yellow_line = np.clip(self.reliability_yellow_line, 0, 100)

        msg = UInt8()
        msg.data = int(self.reliability_yellow_line)
        self.pub_yellow_line_reliability.publish(msg)

        return fraction_num, mask

    def fit_from_lines(self, lane_fit, image):
        nonzeroy, nonzerox = image.nonzero()
        margin = 100
        lane_inds = (
            (nonzerox > lane_fit[0]*nonzeroy**2 + lane_fit[1]*nonzeroy + lane_fit[2] - margin) &
            (nonzerox < lane_fit[0]*nonzeroy**2 + lane_fit[1]*nonzeroy + lane_fit[2] + margin)
        )
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        lane_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
        return lane_fitx, lane_fit

    def sliding_windown(self, image, side):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        lane_base = np.argmax(histogram[:midpoint]) if side == 'left' else np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20
        window_height = image.shape[0] // nwindows
        nonzeroy, nonzerox = image.nonzero()
        x_current = lane_base
        margin = 50
        minpix = 50
        lane_inds = []

        for window in range(nwindows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            good_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) & (nonzerox < win_x_high)
            ).nonzero()[0]

            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except Exception:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
        return lane_fitx, lane_fit

    def make_lane(self, image, yellow_fraction):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_state = UInt8()
        centerx = np.array([image.shape[1] / 2] * image.shape[0])
        self.is_center_x_exist = False

        if yellow_fraction > 3000:
            centerx = self.left_fitx + self.yellow_offset
            self.is_center_x_exist = True
            lane_state.data = 1
        else:
            lane_state.data = 0

        self.pub_lane_state.publish(lane_state)

        if self.is_center_x_exist:
            cx_val = float(centerx.item(350))
            self.get_logger().info(f'[LINE DETECTED] Center X at Y=350: {cx_val:.2f}, Lane State: {lane_state.data}')
            msg = Float64()
            msg.data = cx_val
            self.pub_lane.publish(msg)
        else:
            self.get_logger().info(f'[NO LINE] Lane State: {lane_state.data}')

        if self.pub_image_type == 'compressed':
            color_image = image.copy()

            for y, lx in enumerate(self.left_fitx.astype(np.int32)):
                if 0 <= y < image.shape[0] and 0 <= lx < image.shape[1]:
                    cv2.circle(color_image, (lx, y), 2, (0, 255, 255), -1)  # Yellow

            if self.is_center_x_exist:
                centerx_int = (self.left_fitx + self.yellow_offset).astype(np.int32)
                for y, cx in enumerate(centerx_int):
                    if 0 <= y < image.shape[0] and 0 <= cx < image.shape[1]:
                        cv2.circle(color_image, (cx, y), 2, (0, 0, 255), -1)  # Red

                y_text = 350
                x_left = int(self.left_fitx[y_text])
                x_center = int(centerx_int[y_text])
                cv2.putText(color_image, 'left', (x_left - 30, y_text - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(color_image, 'center', (x_center - 40, y_text + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(color_image, 'jpg'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# ##################################################################################################  노란선만 추종

#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import message_filters
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8

class DetectLane(Node):
    def __init__(self):
        super().__init__('detect_lane')
        self.cvBridge = CvBridge()

        # =================================================================
        # 파라미터 선언 (필요 시 YAML 파일로부터 로드)
        # 예: self.declare_parameter('some_param', 1.0)
        # =================================================================
        
        # Bird's-Eye View 변환 매트릭스 생성
        self.M, self.Minv = self._create_perspective_transform()

        # 멤버 변수 초기화 (이전 프레임의 차선 정보를 저장하기 위함)
        self.left_fit = None
        self.right_fit = None

        # =================================================================
        # 구독 (Subscriber) - 3개의 토픽을 message_filters로 동기화
        # =================================================================
        sub_compensated = message_filters.Subscriber(self, CompressedImage, '/image_compensated/compressed')
        sub_white_mask = message_filters.Subscriber(self, CompressedImage, '/image_compensated/white_mask/compressed')
        sub_yellow_mask = message_filters.Subscriber(self, CompressedImage, '/image_compensated/yellow_mask/compressed')

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [sub_compensated, sub_white_mask, sub_yellow_mask],
            queue_size=10,
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.sync_callback)

        # =================================================================
        # 발행 (Publisher) - 최종 결과물들을 발행할 토픽들
        # =================================================================
        self.pub_image_lane = self.create_publisher(CompressedImage, '/image_lane_detected/compressed', 10)
        self.pub_lane_state = self.create_publisher(UInt8, '/lane_state', 10)
        self.pub_lane_center = self.create_publisher(Float64, '/lane_center', 10)
        self.pub_bev_compensated = self.create_publisher(CompressedImage, '/image_bev_compensated/compressed', 10)
        self.pub_bev_white = self.create_publisher(CompressedImage, '/image_bev_white/compressed', 10)
        self.pub_bev_yellow = self.create_publisher(CompressedImage, '/image_bev_yellow/compressed', 10)
        self.pub_bev_roi = self.create_publisher(CompressedImage, '/image_bev_roi/compressed', 10) 

        self.get_logger().info("DetectLane node has been initialized.")

    def _create_perspective_transform(self):
        """
        src, dst 좌표는 차량의 카메라 설정에 맞게 반드시 튜닝해야 합니다.
        원근 변환을 위한 매트릭스를 생성합니다.
        """
        # 이미지 크기: 1280x720 기준
        img_size = (1280, 720)
        # 원본 이미지의 차선 영역 4개 꼭짓점 (Source points)
        src = np.float32([
        (180, 400),
        (70, 720),
        (1230, 720),
        (1140, 400)
        ])
        # src = np.float32([
        # (460, 360),  # 왼쪽 위
        # (340, 720),  # 왼쪽 아래
        # (840, 720),  # 오른쪽 아래        
        # (720, 360)  # 오른쪽 위
        # ])

        # 변환 후 결과 이미지의 4개 꼭짓점 (Destination points)
        dst = np.float32([
            (0, 0),
            (0, img_size[1]),
            (img_size[0], img_size[1]),
            (img_size[0], 0)
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def sync_callback(self, msg_compensated, msg_white, msg_yellow):
        """메인 콜백 함수: 모든 처리가 여기서 시작됩니다."""

        # =================================================================
        # 아래 info 로그 한 줄을 추가하여 수신 성공 여부를 확인합니다.
        # self.get_logger().info('<<< Successfully received a synchronized set of 3 messages! Starting lane detection... >>>')
        # =================================================================
        # self.get_logger().info(f"img_compensated shape: {img_compensated.shape}")  # 예: (720, 1280, 3)

        # 1. 메시지 디코딩
        # PNG 자체가 압축 포맷이고, 압축된 1채널 이미지는 encoding 정보를 무시하는 경우가 있음.
        # 수정) mono8 명시 => 안정적으로 기본값 사용 => sliding_window 함수 실행 불가
        # sliding_window 알고리즘은 1채널 이미지를 기준으로 만들어졌기 때문에, 3채널을 받아
        # 차선 찾기 실패, /lane_state가 출력x.
        img_compensated = self.cvBridge.compressed_imgmsg_to_cv2(msg_compensated)

        # cv_bridge가 가장 잘 처리하는 기본값(bgr8)으로 먼저 디코딩
        # 결과로 나온 3채널 이미지를 우리가 필요한 1채널 흑백 이미지로 직접 변환
        mask_white = self.cvBridge.compressed_imgmsg_to_cv2(msg_white)
        # 만약 3채널이라면, 1채널로 변환하여 자기 자신에게 다시 덮어씁니다.
        if len(mask_white.shape) == 3:
            mask_white = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
        # 노란색 마스크도 동일하게 처리합니다.
        mask_yellow = self.cvBridge.compressed_imgmsg_to_cv2(msg_yellow)
        if len(mask_yellow.shape) == 3:
            mask_yellow = cv2.cvtColor(mask_yellow, cv2.COLOR_BGR2GRAY)
        #######
        

        # 빨간 테두리 그리기 (원본 이미지 복사본에)
        img_roi = img_compensated.copy()
        pts = np.array([
        [180, 400],  # 왼쪽 위
        [70, 720],  # 왼쪽 아래
        [1230, 720],  # 오른쪽 아래     
        [1140, 400]   # 오른쪽 위
        ], np.int32)
        # pts = np.array([
        # [460, 360],  # 왼쪽 위
        # [340, 720],  # 왼쪽 아래
        # [840, 720],  # 오른쪽 아래     
        # [720, 360]   # 오른쪽 위
        # ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_roi, [pts], isClosed=True, color=(0, 0, 255), thickness=5)  # 빨간색( BGR = (0,0,255) ) 테두리


        # 마스크별 픽셀 개수 출력 (디버깅용)
        # white_pixel_count = np.count_nonzero(mask_white)
        # yellow_pixel_count = np.count_nonzero(mask_yellow)
        # self.get_logger().info(f"White mask pixel count: {white_pixel_count}, Yellow mask pixel count: {yellow_pixel_count}")

        # 2. Bird's-Eye View로 원근 변환
        warped_compensated = cv2.warpPerspective(img_compensated, self.M, (1280, 720), flags=cv2.INTER_LINEAR)
        warped_white_mask = cv2.warpPerspective(mask_white, self.M, (1280, 720), flags=cv2.INTER_LINEAR)
        warped_yellow_mask = cv2.warpPerspective(mask_yellow, self.M, (1280, 720), flags=cv2.INTER_LINEAR)
        
        ## BEV 토픽 발행 설정 #######
        bev_compensated_msg = self.cvBridge.cv2_to_compressed_imgmsg(warped_compensated)
        bev_white_msg = self.cvBridge.cv2_to_compressed_imgmsg(warped_white_mask)
        bev_yellow_msg = self.cvBridge.cv2_to_compressed_imgmsg(warped_yellow_mask)
        roi_msg = self.cvBridge.cv2_to_compressed_imgmsg(img_roi)


        self.pub_bev_compensated.publish(bev_compensated_msg)
        self.pub_bev_white.publish(bev_white_msg)
        self.pub_bev_yellow.publish(bev_yellow_msg)
        self.pub_bev_roi.publish(roi_msg)
        ##############################

        # 3. 차선 픽셀 검출 및 2차 다항식으로 피팅
        if self.left_fit is None or self.right_fit is None:
            # 첫 프레임이거나 차선을 놓쳤을 경우, Sliding Window로 처음부터 찾기
            left_fitx, self.left_fit = self.sliding_window(warped_yellow_mask, 'left')
            right_fitx, self.right_fit = self.sliding_window(warped_white_mask, 'right')
        else:
            # 이전 프레임의 차선 위치 주변에서 빠르게 다시 찾기
            left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, warped_yellow_mask)
            right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, warped_white_mask)

        # =================================================================
        # =================================================================
        # [수정된 부분] 여기서 안전장치를 추가합니다.
        # # 차선 감지에 하나라도 실패했다면(None이 반환되었다면), 더 이상 진행하지 않고 현재 프레임 처리를 중단합니다.
        # if self.left_fit is None or self.right_fit is None:
        #     self.get_logger().warn('Lane detection failed, skipping frame.')
            
        #     # 차선을 못 찾았을 경우, 보정된 원본 이미지만이라도 발행하여 상태를 확인할 수 있습니다.
        #     # (이 줄의 주석을 해제하면 됩니다.)
        #     self.pub_image_lane.publish(msg_compensated)
        #     return
        # =================================================================

        # 4. 결과 그리기 및 최종 정보 발행 (안전장치를 통과한 경우에만 실행됨)
        self.draw_lane_and_publish(img_compensated, warped_yellow_mask, warped_white_mask, left_fitx, right_fitx)

    def sliding_window(self, img_w, left_or_right):
        """Sliding Window 알고리즘으로 차선을 처음부터 찾습니다."""
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        
        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        else: # 'right'
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows, margin, minpix = 20, 50, 50
        window_height = int(img_w.shape[0] / nwindows)
        nonzero = img_w.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        x_current = lane_base
        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            
            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))
                
        lane_inds = np.concatenate(lane_inds)
        x, y = nonzerox[lane_inds], nonzeroy[lane_inds]
    
     # [디버깅 로그 추가] 슬라이딩 윈도우를 통해 최종적으로 몇 개의 픽셀이 선택되었는지 출력합니다.
        self.get_logger().info(f"[{left_or_right.upper()}] Pixels selected by sliding window: {len(x)}")
    
        # [수정된 부분 시작]
        # 픽셀이 충분히 감지되었는지 확인
        if len(x) < minpix * 3: # 안정적인 피팅을 위해 최소 픽셀 수 조건 강화
            # 픽셀이 부족하면 이전 프레임의 값을 사용하거나, 이전 값도 없으면 None 반환
            previous_fit = self.left_fit if left_or_right == 'left' else self.right_fit
            if previous_fit is None:
                return None, None # fitted_x와 lane_fit 모두 None으로 반환
            else:
                lane_fit = previous_fit
        else:
            try:
                lane_fit = np.polyfit(y, x, 2)
            except TypeError:
                # polyfit이 실패할 경우에도 대비
                previous_fit = self.left_fit if left_or_right == 'left' else self.right_fit
                if previous_fit is None:
                    return None, None
                lane_fit = previous_fit
        # [수정된 부분 끝]
                
        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        fitted_x = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
        
        return fitted_x, lane_fit

    def fit_from_lines(self, lane_fit, image):
        """이전 차선 위치 주변에서 새로운 차선을 찾습니다."""
        nonzero = image.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        margin = 100
        
        lane_inds = ((nonzerox > (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
                     (nonzerox < (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin)))
        
        x, y = nonzerox[lane_inds], nonzeroy[lane_inds]

        try:
            new_fit = np.polyfit(y, x, 2)
        except TypeError:
            new_fit = lane_fit

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        fitted_x = new_fit[0] * ploty ** 2 + new_fit[1] * ploty + new_fit[2]
        
        return fitted_x, new_fit

    def draw_lane_and_publish(self, base_image, yellow_mask, white_mask, left_fitx, right_fitx):
            """
            [수정됨] 결과를 시각화하고 최종 정보를 안정적으로 발행합니다.
            계산, 시각화, 발행의 각 단계를 명확히 분리하고 None 체크를 강화했습니다.
            """
            # self.get_logger().info("--- Entering draw_lane_and_publish ---")

            # 1. 초기 설정 및 변수 준비
            ploty = np.linspace(0, base_image.shape[0] - 1, base_image.shape[0])
            warp_zero = np.zeros_like(yellow_mask).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # 2. 차선 상태 및 중앙선(centerx) 계산
            # 이 단계에서는 아직 그림을 그리지 않고, 필요한 값만 계산합니다.
            
            # 차선 검출 여부 확인
            yellow_detected = np.count_nonzero(yellow_mask) > 3000
            white_detected = np.count_nonzero(white_mask) > 3000

            centerx = None  # 중앙선 좌표 배열, 계산 실패 시 None 유지

            # [개선] 중앙선 계산 로직을 명확하게 정리
            # 이상적인 경우: 양쪽 차선 모두 감지
            if left_fitx is not None and right_fitx is not None:
                self.get_logger().info("Calculating center from BOTH lanes.")
                # 두 배열의 길이가 다를 수 있으므로, 짧은 쪽에 맞춰서 안전하게 계산
                min_len = min(len(left_fitx), len(right_fitx))
                centerx = np.mean([left_fitx[:min_len], right_fitx[:min_len]], axis=0)

            # 차선 한쪽만 감지된 경우: 감지된 차선에서 일정 거리(320px)를 오프셋하여 중앙선 추정
            elif left_fitx is not None:
                self.get_logger().info("Estimating center from LEFT lane only.")
                centerx = right_fitx - 320
            elif right_fitx is not None:
                self.get_logger().info("Estimating center from RIGHT lane only.")
                centerx = left_fitx + 320
            else:
                # 양쪽 차선 모두 감지 실패. centerx는 그대로 None.
                self.get_logger().warn("Cannot calculate center, both lanes are missing.")


            # 3. 차선 시각화 (BEV 이미지에 그리기)
            # 이 단계에서는 계산된 값을 바탕으로 그림만 그립니다.
            # [핵심 수정] 모든 그리기 작업 전에 None 여부를 반드시 확인합니다.

            # 가. 왼쪽(노란색) 차선 그리기
            if left_fitx is not None:
                # ploty 배열도 left_fitx 길이에 맞춰 잘라줍니다.
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty[:len(left_fitx)]]))])
                cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 255, 0), thickness=25)

            # 나. 오른쪽(흰색) 차선 그리기
            if right_fitx is not None:
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty[:len(right_fitx)]]))])
                cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)

            # 다. 중앙 주행 가능 영역 채우기 (양쪽 차선이 모두 있을 때만)
            if left_fitx is not None and right_fitx is not None:
                min_len = min(len(left_fitx), len(right_fitx))
                pts_left = np.array([np.transpose(np.vstack([left_fitx[:min_len], ploty[:min_len]]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[:min_len], ploty[:min_len]])))])
                pts = np.hstack((pts_left, pts_right))
                cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
                
            # 라. 중앙선(추정선 포함) 그리기
            if centerx is not None:
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty[:len(centerx)]]))])
                cv2.polylines(color_warp, np.int32([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
            
            # self.get_logger().info("Step C: Drawing on BEV image complete.")


            # 4. 최종 이미지 생성 및 정보 발행
            # 원근 복원 및 합성
            unwarped_lane = cv2.warpPerspective(color_warp, self.Minv, (base_image.shape[1], base_image.shape[0]))
            final_image = cv2.addWeighted(base_image, 1, unwarped_lane, 0.5, 0)
            # self.get_logger().info("Step D: Un-warping and composition complete.")

            # 최종 이미지 퍼블리시
            final_image_msg = self.cvBridge.cv2_to_compressed_imgmsg(final_image, 'png')
            self.pub_image_lane.publish(final_image_msg)
            # self.get_logger().info("Step E: Published final detected image.")

            # 차선 상태(lane_state) 퍼블리시
            lane_state = 0
            if yellow_detected and white_detected: lane_state = 2
            elif yellow_detected: lane_state = 1
            elif white_detected: lane_state = 3
            
            lane_state_msg = UInt8(data=lane_state)
            self.pub_lane_state.publish(lane_state_msg)
            # self.get_logger().info(f"Step F: Published lane state: {lane_state}")

            # 중앙 위치 퍼블리시 (centerx가 성공적으로 계산되었을 경우에만)
            if centerx is not None:
                if len(centerx) > 400:  # 로봇 앞 일정 거리의 차선 중앙값
                    desired_center = centerx[400]
                    self.pub_lane_center.publish(Float64(data=desired_center))
                    # self.get_logger().info("Step G: Published lane center.")
                else:
                    self.get_logger().warn(f"centerx length ({len(centerx)}) is too short, skipping publish.")
            
            # self.get_logger().info("--- Leaving draw_lane_and_publish ---")

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

###################################################################################################  이진화된이미지


import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.sub_image_original = self.create_subscription(
            CompressedImage,
            '/camera/preprocessed/compressed',
            self.cbFindLane,
            1
        )

        self.pub_image_lane = self.create_publisher(CompressedImage, '/detect/image_masked/compressed', 1)
        self.pub_image_output = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()
        self.counter = 1
        self.left_fit = np.array([0., 0., 300.])
        self.right_fit = np.array([0., 0., 980.])
        self.lane_fit_bef = np.array([0., 0., 0.])

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))
        self.mov_avg_length = 5
        self.left_fitx = np.array([])
        self.right_fitx = np.array([])

        self.detection_threshold = 3000

    def cbFindLane(self, msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        height, width = binary.shape
        left_mask = np.zeros_like(binary)
        right_mask = np.zeros_like(binary)
        margin = 50
        left_mask[:, :width//2 + margin] = binary[:, :width//2 + margin]
        right_mask[:, width//2 - margin:] = binary[:, width//2 - margin:]

        ploty = np.linspace(0, height - 1, height)

        if np.count_nonzero(left_mask) > self.detection_threshold:
            self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, left_mask)
            self.mov_avg_left = np.append(self.mov_avg_left, np.array([self.left_fit]), axis=0)

        if np.count_nonzero(right_mask) > self.detection_threshold:
            self.right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, right_mask)
            self.mov_avg_right = np.append(self.mov_avg_right, np.array([self.right_fit]), axis=0)

        if self.mov_avg_left.shape[0] > 0:
            self.left_fit = np.mean(self.mov_avg_left[-self.mov_avg_length:], axis=0)
            self.left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]

        if self.mov_avg_right.shape[0] > 0:
            self.right_fit = np.mean(self.mov_avg_right[-self.mov_avg_length:], axis=0)
            self.right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        half_width = 400
        centerx = (self.left_fitx + self.right_fitx) / 2 if self.left_fitx.size > 0 and self.right_fitx.size > 0 else \
                  self.left_fitx + half_width if self.left_fitx.size > 0 else \
                  self.right_fitx - half_width if self.right_fitx.size > 0 else \
                  np.array([width / 2] * height)

        cx = float(centerx[height//2])
        self.pub_lane.publish(Float64(data=cx))

        lane_state = UInt8()
        lane_state.data = 2 if self.left_fitx.size > 0 and self.right_fitx.size > 0 else \
                          1 if self.left_fitx.size > 0 else \
                          3 if self.right_fitx.size > 0 else 0
        self.pub_lane_state.publish(lane_state)

        self.get_logger().info(f"LaneState: {lane_state.data}, CenterX: {cx:.2f}")

        color_warp = np.zeros_like(cv_image)
        if self.left_fitx.size > 0:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 0, 255), thickness=10)
            y_l = int(height * 0.6)
            x_l = int(self.left_fitx[y_l])
            cv2.putText(color_warp, 'L', (x_l - 10, y_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        if self.right_fitx.size > 0:
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 255, 0), thickness=10)
            y_r = int(height * 0.6)
            x_r = int(self.right_fitx[y_r])
            cv2.putText(color_warp, 'R', (x_r - 10, y_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

        if self.left_fitx.size > 0 and self.right_fitx.size > 0:
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(color_warp, np.int32(pts_center), isClosed=False, color=(255, 0, 0), thickness=2)
            mid_y = int(height * 0.5)
            mid_x = int(centerx[mid_y])
            cv2.putText(color_warp, 'CENTER', (mid_x - 40, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        final = cv2.addWeighted(cv_image, 1, color_warp, 0.6, 0)
        self.pub_image_output.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))
        color_mask = cv2.bitwise_and(cv_image, cv_image, mask=binary)
        self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(color_mask, 'jpg'))

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        y, x = nonzero[0], nonzero[1]
        lane_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
        return lane_fitx, lane_fit


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


###################################################################################################  마스크만 합치고 차선검출은 따로

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.sub_image_original = self.create_subscription(
            CompressedImage,
            '/camera/preprocessed/compressed',
            self.cbFindLane,
            1
        )

        self.pub_image_lane = self.create_publisher(CompressedImage, '/detect/image_masked/compressed', 1)
        self.pub_image_output = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)
        self.pub_left_line_reliability = self.create_publisher(UInt8, '/detect/left_line_reliability', 1)
        self.pub_right_line_reliability = self.create_publisher(UInt8, '/detect/right_line_reliability', 1)

        self.cvBridge = CvBridge()
        self.counter = 1
        self.reliability_left_line = 100
        self.reliability_right_line = 100
        self.left_fit = np.array([0., 0., 300.])
        self.right_fit = np.array([0., 0., 980.])
        self.lane_fit_bef = np.array([0., 0., 0.])

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))
        self.mov_avg_length = 5
        self.left_fitx = np.array([])
        self.right_fitx = np.array([])

        # 파라미터화된 값들
        self.reliability_threshold = 100
        self.reliability_step = 5
        self.detection_threshold = 3000

    def compute_reliability(self, mask, current_reliability):
        height = mask.shape[0]
        how_much_short = height - np.count_nonzero(np.any(mask > 0, axis=1))
        if how_much_short > self.reliability_threshold:
            current_reliability = max(0, current_reliability - self.reliability_step)
        else:
            current_reliability = min(100, current_reliability + self.reliability_step)
        return current_reliability

    def cbFindLane(self, msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        white_hsv_min = np.array([0, 0, 100])
        white_hsv_max = np.array([180, 150, 255])
        yellow_hsv_min = np.array([18, 40, 100])
        yellow_hsv_max = np.array([45, 255, 255])

        mask_white = cv2.inRange(hsv, white_hsv_min, white_hsv_max)
        mask_yellow = cv2.inRange(hsv, yellow_hsv_min, yellow_hsv_max)
        mask = cv2.bitwise_or(mask_white, mask_yellow)

        fraction_white = np.count_nonzero(mask_white)
        fraction_yellow = np.count_nonzero(mask_yellow)

        self.reliability_left_line = self.compute_reliability(mask_yellow, self.reliability_left_line)
        self.reliability_right_line = self.compute_reliability(mask_white, self.reliability_right_line)

        self.pub_left_line_reliability.publish(UInt8(data=self.reliability_left_line))
        self.pub_right_line_reliability.publish(UInt8(data=self.reliability_right_line))

        ploty = np.linspace(0, mask.shape[0] - 1, mask.shape[0])
        try:
            if fraction_yellow > self.detection_threshold:
                self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, mask_yellow)
                self.mov_avg_left = np.append(self.mov_avg_left, np.array([self.left_fit]), axis=0)
            if fraction_white > self.detection_threshold:
                self.right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, mask_white)
                self.mov_avg_right = np.append(self.mov_avg_right, np.array([self.right_fit]), axis=0)
        except:
            if fraction_yellow > self.detection_threshold:
                self.left_fitx, self.left_fit = self.sliding_window(mask_yellow, 'left')
                self.mov_avg_left = np.array([self.left_fit])
            if fraction_white > self.detection_threshold:
                self.right_fitx, self.right_fit = self.sliding_window(mask_white, 'right')
                self.mov_avg_right = np.array([self.right_fit])

        if self.mov_avg_left.shape[0] > 0:
            self.left_fit = np.mean(self.mov_avg_left[-self.mov_avg_length:], axis=0)
            self.left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        if self.mov_avg_right.shape[0] > 0:
            self.right_fit = np.mean(self.mov_avg_right[-self.mov_avg_length:], axis=0)
            self.right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        half_width = 400
        centerx = (self.left_fitx + self.right_fitx) / 2 if fraction_yellow > self.detection_threshold and fraction_white > self.detection_threshold else \
                  self.left_fitx + half_width if fraction_yellow > self.detection_threshold else \
                  self.right_fitx - half_width if fraction_white > self.detection_threshold else \
                  np.array([cv_image.shape[1] / 2] * mask.shape[0])

        self.pub_lane.publish(Float64(data=centerx[350]))

        lane_state = UInt8()
        lane_state.data = 2 if fraction_yellow > self.detection_threshold and fraction_white > self.detection_threshold else \
                          1 if fraction_yellow > self.detection_threshold else \
                          3 if fraction_white > self.detection_threshold else 0
        self.pub_lane_state.publish(lane_state)

        self.get_logger().info(f"LaneState: {lane_state.data}, CenterX: {centerx[350]:.2f}")

        color_warp = np.zeros_like(cv_image)
        if self.left_fitx.size > 0 and self.right_fitx.size > 0:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            pts = np.hstack((pts_left, pts_right))

            if fraction_yellow > self.detection_threshold:
                cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 0, 255), thickness=10)
            if fraction_white > self.detection_threshold:
                cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 255, 0), thickness=10)
            if fraction_yellow > self.detection_threshold and fraction_white > self.detection_threshold:
                cv2.fillPoly(color_warp, np.int32([pts]), color=(0, 255, 0))

        if centerx is not None and centerx.size == ploty.size:
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(color_warp, np.int32(pts_center), isClosed=False, color=(255, 0, 0), thickness=2)

            mid_y = int(mask.shape[0] * 0.5)
            mid_x = int(centerx[mid_y])
            cv2.putText(color_warp, 'CENTER', (mid_x - 40, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        if self.left_fitx.size > 0:
            y_l = int(mask.shape[0] * 0.6)
            x_l = int(self.left_fitx[y_l])
            cv2.putText(color_warp, 'L', (x_l - 10, y_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        if self.right_fitx.size > 0:
            y_r = int(mask.shape[0] * 0.6)
            x_r = int(self.right_fitx[y_r])
            cv2.putText(color_warp, 'R', (x_r - 10, y_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        final = cv2.addWeighted(cv_image, 1, color_warp, 0.6, 0)
        self.pub_image_output.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

        color_mask = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(color_mask, 'jpg'))

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        y, x = nonzero[0], nonzero[1]
        lane_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
        return lane_fitx, lane_fit

    def sliding_window(self, img_w, side):
        histogram = np.sum(img_w[int(img_w.shape[0]/2):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        lane_base = np.argmax(histogram[:midpoint]) if side == 'left' else np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 20
        window_height = int(img_w.shape[0] / nwindows)
        nonzero = img_w.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        x_current = lane_base
        margin = 50
        minpix = 50
        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
        return lane_fitx, lane_fit


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


###################################################################################################  yaml받아서 범위 합친거

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from std_msgs.msg import UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.declare_parameter('raw_mode', False)
        self.raw_mode = self.get_parameter('raw_mode').get_parameter_value().bool_value

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(from_value=0, to_value=179, step=1)]
        )
        parameter_descriptor_sl = ParameterDescriptor(
            description='saturation and lightness range',
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)]
        )

        self.declare_parameters(
            namespace='',
            parameters=[
                ('detect.lane.white.hue_l', 0, parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179, parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 10, parameter_descriptor_sl),
                ('detect.lane.white.saturation_h', 60, parameter_descriptor_sl),
                ('detect.lane.white.lightness_l', 180, parameter_descriptor_sl),
                ('detect.lane.white.lightness_h', 255, parameter_descriptor_sl),
                ('detect.lane.yellow.hue_l', 20, parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 40, parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 50, parameter_descriptor_sl),
                ('detect.lane.yellow.saturation_h', 255, parameter_descriptor_sl),
                ('detect.lane.yellow.lightness_l', 60, parameter_descriptor_sl),
                ('detect.lane.yellow.lightness_h', 255, parameter_descriptor_sl),
                ('is_detection_calibration_mode', False)
            ]
        )

        self.hue_white_l = self.get_parameter('detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter('detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter('detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter('detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = self.get_parameter('detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = self.get_parameter('detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter('detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter('detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter('detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter('detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter('detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter('detect.lane.yellow.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter('is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cb_calibration_param)

        self.sub_image_type = 'compressed'
        self.pub_image_type = 'compressed'

        subscribe_topic = '/camera/image_raw/compressed' if self.raw_mode else '/camera/preprocessed/compressed'
        self.sub_image_original = self.create_subscription(
            CompressedImage, subscribe_topic, self.cbFindLane, 1
        )

        self.pub_image_lane = self.create_publisher(CompressedImage, '/detect/image_masked/compressed', 1)
        self.pub_image_output = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)
        self.pub_left_line_reliability = self.create_publisher(UInt8, '/detect/left_line_reliability', 1)
        self.pub_right_line_reliability = self.create_publisher(UInt8, '/detect/right_line_reliability', 1)

        self.cvBridge = CvBridge()
        self.counter = 1
        self.reliability_left_line = 100
        self.reliability_right_line = 100
        self.left_fit = np.array([0., 0., 300.])
        self.right_fit = np.array([0., 0., 980.])
        self.lane_fit_bef = np.array([0., 0., 0.])

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))
        self.mov_avg_length = 5
        self.left_fitx = np.array([])
        self.right_fitx = np.array([])

    def cb_calibration_param(self, parameters):
        for param in parameters:
            if hasattr(self, param.name.split('.')[-1]):
                setattr(self, param.name.split('.')[-1], param.value)
        return SetParametersResult(successful=True)

    def compute_reliability(self, mask, current_reliability):
        height = mask.shape[0]
        how_much_short = height - np.count_nonzero(np.any(mask > 0, axis=1))
        if how_much_short > 100:
            current_reliability = max(0, current_reliability - 5)
        else:
            current_reliability = min(100, current_reliability + 5)
        return current_reliability

    def cbFindLane(self, msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, (self.hue_white_l, self.saturation_white_l, self.lightness_white_l),
                                      (self.hue_white_h, self.saturation_white_h, self.lightness_white_h))
        mask_yellow = cv2.inRange(hsv, (self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l),
                                       (self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h))
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        fraction_white = np.count_nonzero(mask_white)
        fraction_yellow = np.count_nonzero(mask_yellow)

        self.reliability_left_line = self.compute_reliability(mask_yellow, self.reliability_left_line)
        self.reliability_right_line = self.compute_reliability(mask_white, self.reliability_right_line)

        self.pub_left_line_reliability.publish(UInt8(data=self.reliability_left_line))
        self.pub_right_line_reliability.publish(UInt8(data=self.reliability_right_line))

        ploty = np.linspace(0, mask.shape[0] - 1, mask.shape[0])
        try:
            if fraction_yellow > 3000:
                self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, mask_yellow)
                self.mov_avg_left = np.append(self.mov_avg_left, np.array([self.left_fit]), axis=0)
            if fraction_white > 3000:
                self.right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, mask_white)
                self.mov_avg_right = np.append(self.mov_avg_right, np.array([self.right_fit]), axis=0)
        except:
            if fraction_yellow > 3000:
                self.left_fitx, self.left_fit = self.sliding_window(mask_yellow, 'left')
                self.mov_avg_left = np.array([self.left_fit])
            if fraction_white > 3000:
                self.right_fitx, self.right_fit = self.sliding_window(mask_white, 'right')
                self.mov_avg_right = np.array([self.right_fit])

        if self.mov_avg_left.shape[0] > 0:
            self.left_fit = np.mean(self.mov_avg_left[-self.mov_avg_length:], axis=0)
            self.left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        if self.mov_avg_right.shape[0] > 0:
            self.right_fit = np.mean(self.mov_avg_right[-self.mov_avg_length:], axis=0)
            self.right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        centerx = (self.left_fitx + self.right_fitx) / 2 if fraction_yellow > 3000 and fraction_white > 3000 else \
                  self.left_fitx + 280 if fraction_yellow > 3000 else \
                  self.right_fitx - 280 if fraction_white > 3000 else \
                  np.array([cv_image.shape[1] / 2] * mask.shape[0])

        self.pub_lane.publish(Float64(data=centerx[350]))

        lane_state = UInt8()
        lane_state.data = 2 if fraction_yellow > 3000 and fraction_white > 3000 else \
                          1 if fraction_yellow > 3000 else \
                          3 if fraction_white > 3000 else 0
        self.pub_lane_state.publish(lane_state)

        self.get_logger().info(f"LaneState: {lane_state.data}, CenterX: {centerx[350]:.2f}")

        color_warp = np.zeros_like(cv_image)
        if self.left_fitx.size > 0 and self.right_fitx.size > 0:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            pts = np.hstack((pts_left, pts_right))

            if fraction_yellow > 3000:
                cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0, 0, 255), thickness=10)
            if fraction_white > 3000:
                cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255, 255, 0), thickness=10)
            if fraction_yellow > 3000 and fraction_white > 3000:
                cv2.fillPoly(color_warp, np.int32([pts]), color=(0, 255, 0))

        final = cv2.addWeighted(cv_image, 1, color_warp, 0.6, 0)

        self.pub_image_output.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

        color_mask = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(color_mask, 'jpg'))

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        y, x = nonzero[0], nonzero[1]
        lane_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
        return lane_fitx, lane_fit

    def sliding_window(self, img_w, side):
        histogram = np.sum(img_w[int(img_w.shape[0]/2):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        lane_base = np.argmax(histogram[:midpoint]) if side == 'left' else np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 20
        window_height = int(img_w.shape[0] / nwindows)
        nonzero = img_w.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        x_current = lane_base
        margin = 50
        minpix = 50
        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
        return lane_fitx, lane_fit

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



###########################################################################33############## 토픽바꾸고 나머지 미완

#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import IntegerRange, ParameterDescriptor
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, UInt8
from cv_bridge import CvBridge

class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane2')


        hue_desc = ParameterDescriptor(
            description='hue range',
            integer_range=[IntegerRange(from_value=0, to_value=179, step=1)]
        )
        sat_light_desc = ParameterDescriptor(
            description='saturation/lightness range',
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)]
)


        self.declare_parameters('', [
            ('detect.lane.white.hue_l', 0, hue_desc),
            ('detect.lane.white.hue_h', 179, hue_desc),
            ('detect.lane.white.saturation_l', 10, sat_light_desc),
            ('detect.lane.white.saturation_h', 60, sat_light_desc),
            ('detect.lane.white.lightness_l', 180, sat_light_desc),
            ('detect.lane.white.lightness_h', 255, sat_light_desc),
            ('detect.lane.yellow.hue_l', 20, hue_desc),
            ('detect.lane.yellow.hue_h', 40, hue_desc),
            ('detect.lane.yellow.saturation_l', 50, sat_light_desc),
            ('detect.lane.yellow.saturation_h', 255, sat_light_desc),
            ('detect.lane.yellow.lightness_l', 60, sat_light_desc),
            ('detect.lane.yellow.lightness_h', 255, sat_light_desc),
        ])

        self.load_parameters()
        self.bridge = CvBridge()

        # 구독자
        self.sub_image = self.create_subscription(
            CompressedImage, '/camera/preprocessed/compressed', self.cbFindLane, 1)

        # 퍼블리셔
        self.pub_image_lane = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)
        self.pub_white_line_reliability = self.create_publisher(UInt8, '/detect/white_line_reliability', 1)
        self.pub_yellow_line_reliability = self.create_publisher(UInt8, '/detect/yellow_line_reliability', 1)
        self.pub_white_mask = self.create_publisher(CompressedImage, '/detect/image_output_white/compressed', 1)
        self.pub_yellow_mask = self.create_publisher(CompressedImage, '/detect/image_output_yellow/compressed', 1)

        self.counter = 1

    def load_parameters(self):
        p = self.get_parameter
        self.hue_white_l = p('detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = p('detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = p('detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = p('detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = p('detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = p('detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = p('detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = p('detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = p('detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = p('detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = p('detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = p('detect.lane.yellow.lightness_h').get_parameter_value().integer_value

    def cbFindLane(self, msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.process_lane(cv_image, msg.header)

    def process_lane(self, img, header):
        white_mask = self.get_mask(img, 'white')
        yellow_mask = self.get_mask(img, 'yellow')

        final, state, center_x = self.make_lane(img, white_mask, yellow_mask)

        self.pub_image_lane.publish(self.bridge.cv2_to_compressed_imgmsg(final, dst_format='jpg'))
        self.pub_lane_state.publish(UInt8(data=state))
        if center_x is not None:
            self.pub_lane.publish(Float64(data=float(center_x)))
            self.get_logger().info(f"CenterX: {center_x}")

    def get_mask(self, img, color):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color == 'white':
            lower = np.array([self.hue_white_l, self.saturation_white_l, self.lightness_white_l])
            upper = np.array([self.hue_white_h, self.saturation_white_h, self.lightness_white_h])
        else:
            lower = np.array([self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l])
            upper = np.array([self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h])

        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = np.count_nonzero(mask)
        reliability = min(int(pixel_count / 1000), 100)

        if color == 'white':
            self.pub_white_line_reliability.publish(UInt8(data=reliability))
            self.pub_white_mask.publish(self.bridge.cv2_to_compressed_imgmsg(mask, dst_format='jpg'))
        else:
            self.pub_yellow_line_reliability.publish(UInt8(data=reliability))
            self.pub_yellow_mask.publish(self.bridge.cv2_to_compressed_imgmsg(mask, dst_format='jpg'))

        return mask

    def sliding_window(self, mask, side):
        histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
        midpoint = histogram.shape[0] // 2
        base = np.argmax(histogram[:midpoint]) if side == 'left' else np.argmax(histogram[midpoint:]) + midpoint

        nwindows, margin, minpix = 20, 50, 50
        window_height = mask.shape[0] // nwindows
        nonzero = mask.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        x_current = base
        lane_inds = []

        for window in range(nwindows):
            win_y_low = mask.shape[0] - (window + 1) * window_height
            win_y_high = mask.shape[0] - window * window_height
            win_x_low, win_x_high = x_current - margin, x_current + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x, y = nonzerox[lane_inds], nonzeroy[lane_inds]
        fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        return fitx, ploty

    def make_lane(self, img, white_mask, yellow_mask):
        yellow_exist = np.count_nonzero(yellow_mask) > 3000
        white_exist = np.count_nonzero(white_mask) > 3000
        warp = np.zeros_like(img)
        center_x = None

        if yellow_exist:
            left_fitx, ploty = self.sliding_window(yellow_mask, 'left')
            pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
            cv2.polylines(warp, np.int32([pts_left]), False, (0,0,255), 25)

        if white_exist:
            right_fitx, ploty = self.sliding_window(white_mask, 'right')
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(warp, np.int32([pts_right]), False, (255,255,0), 25)

        state = 0

        if white_exist and yellow_exist:
            centerx = (left_fitx + right_fitx) / 2
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(warp, np.int32([pts_center]), False, (0,255,255), 12)
            state = 2
            center_x = centerx[350]
        elif yellow_exist:
            centerx = left_fitx + 280
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(warp, np.int32([pts_center]), False, (0,255,255), 12)
            state = 1
            center_x = centerx[350]
        elif white_exist:
            centerx = right_fitx - 280
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(warp, np.int32([pts_center]), False, (0,255,255), 12)
            state = 3
            center_x = centerx[350]

        final = cv2.addWeighted(img, 1, warp, 1, 0)
        return final, state, center_x

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
