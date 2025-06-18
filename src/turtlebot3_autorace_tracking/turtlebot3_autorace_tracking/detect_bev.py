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
        원근 변환을 위한 매트릭스를 생성합니다.
        src, dst 좌표는 차량의 카메라 설정에 맞게 반드시 튜닝해야 합니다.
        """
        # 이미지 크기: 1280x720 기준
        img_size = (1280, 720)
        # 원본 이미지의 차선 영역 4개 꼭짓점 (Source points)
        src = np.float32([
        (340, 720),  # 왼쪽 아래
        (840, 720),  # 오른쪽 아래      4    3
        (720, 360),  # 오른쪽 위        1    2
        (460, 360)   # 왼쪽 위
        ])

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
        # 수정) mono8 명시 => 안정적으로 기본값 사용
        img_compensated = self.cvBridge.compressed_imgmsg_to_cv2(msg_compensated)
        mask_white = self.cvBridge.compressed_imgmsg_to_cv2(msg_white)
        mask_yellow = self.cvBridge.compressed_imgmsg_to_cv2(msg_yellow)

        # 빨간 테두리 그리기 (원본 이미지 복사본에)
        img_roi = img_compensated.copy()
        pts = np.array([
            [340, 720],  # 왼쪽 아래
            [840, 720],   # 오른쪽 아래
            [720, 360],  # 오른쪽 위
            [460, 360]   # 왼쪽 위
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_roi, [pts], isClosed=True, color=(0, 0, 255), thickness=5)  # 빨간색( BGR = (0,0,255) ) 테두리


        # 마스크별 픽셀 개수 출력 (디버깅용)
        white_pixel_count = np.count_nonzero(mask_white)
        yellow_pixel_count = np.count_nonzero(mask_yellow)
        self.get_logger().info(f"White mask pixel count: {white_pixel_count}, Yellow mask pixel count: {yellow_pixel_count}")

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
        # [수정된 부분] 여기서 안전장치를 추가합니다.
        # =================================================================
        # 차선 감지에 하나라도 실패했다면(None이 반환되었다면), 더 이상 진행하지 않고 현재 프레임 처리를 중단합니다.
        if self.left_fit is None or self.right_fit is None:
            self.get_logger().warn('Lane detection failed, skipping frame.')
            
            # 차선을 못 찾았을 경우, 보정된 원본 이미지만이라도 발행하여 상태를 확인할 수 있습니다.
            # (이 줄의 주석을 해제하면 됩니다.)
            # self.pub_image_lane.publish(msg_compensated)
            return
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
        """결과를 시각화하고 최종 정보를 발행합니다."""
        ploty = np.linspace(0, base_image.shape[0] - 1, base_image.shape[0])
        
        # 그리기용 빈 이미지 생성
        warp_zero = np.zeros_like(yellow_mask).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # 차선 검출 영역 및 신뢰도 판단
        yellow_detected = np.count_nonzero(yellow_mask) > 3000
        white_detected = np.count_nonzero(white_mask) > 3000

        lane_state = 0
        if yellow_detected and white_detected: lane_state = 2
        elif yellow_detected: lane_state = 1
        elif white_detected: lane_state = 3

        # 중앙선 계산 및 그리기
        centerx = None
        if lane_state == 2:
            centerx = np.mean([left_fitx, right_fitx], axis=0)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0)) # 주행 가능 영역
        elif lane_state == 1:
            centerx = left_fitx + 320 # 차선 폭만큼 이동 (튜닝 필요)
        elif lane_state == 3:
            centerx = right_fitx - 320 # 차선 폭만큼 이동 (튜닝 필요)
            
        if yellow_detected:
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 255, 0), thickness=25)
        if white_detected:
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)
        if centerx is not None:
             pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
             cv2.polylines(color_warp, np.int32([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)

        # 원근을 다시 원래대로 되돌리기
        unwarped_lane = cv2.warpPerspective(color_warp, self.Minv, (base_image.shape[1], base_image.shape[0]))
        # 원본 이미지와 합성
        final_image = cv2.addWeighted(base_image, 1, unwarped_lane, 0.5, 0)
        
        # 최종 정보 발행
        self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final_image))
        self.pub_lane_state.publish(UInt8(data=lane_state))
        if centerx is not None:
            # 차량 제어에 사용할 특정 지점의 중앙값 발행 (예: 이미지의 y=400 지점)
            desired_center = centerx[400]
            self.pub_lane_center.publish(Float64(data=desired_center))

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