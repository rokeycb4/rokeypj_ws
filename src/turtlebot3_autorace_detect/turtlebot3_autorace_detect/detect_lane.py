# #!/usr/bin/env python3
# import cv2
# import numpy as np
# import rclpy
# import message_filters
# from cv_bridge import CvBridge
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from std_msgs.msg import Float64, UInt8

# class DetectLane(Node):
#     def __init__(self):
#         super().__init__('detect_lane')
#         self.cvBridge = CvBridge()

#         # =================================================================
#         # 파라미터 선언 (필요 시 YAML 파일로부터 로드)
#         # 예: self.declare_parameter('some_param', 1.0)
#         # =================================================================
        
#         # Bird's-Eye View 변환 매트릭스 생성
#         self.M, self.Minv = self._create_perspective_transform()

#         # 멤버 변수 초기화 (이전 프레임의 차선 정보를 저장하기 위함)
#         self.left_fit = None
#         self.right_fit = None

#         # =================================================================
#         # 구독 (Subscriber) - 3개의 토픽을 message_filters로 동기화
#         # =================================================================
#         sub_compensated = message_filters.Subscriber(self, CompressedImage, '/image_compensated/compressed')
#         sub_white_mask = message_filters.Subscriber(self, CompressedImage, '/image_compensated/white_mask/compressed')
#         sub_yellow_mask = message_filters.Subscriber(self, CompressedImage, '/image_compensated/yellow_mask/compressed')

#         self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
#             [sub_compensated, sub_white_mask, sub_yellow_mask],
#             queue_size=10,
#             slop=0.1
#         )
#         self.time_synchronizer.registerCallback(self.sync_callback)

#         # =================================================================
#         # 발행 (Publisher) - 최종 결과물들을 발행할 토픽들
#         # =================================================================
#         self.pub_image_lane = self.create_publisher(CompressedImage, '/image_lane_detected/compressed', 10)
#         self.pub_lane_state = self.create_publisher(UInt8, '/lane_state', 10)
#         self.pub_lane_center = self.create_publisher(Float64, '/lane_center', 10)
#         self.pub_bev_compensated = self.create_publisher(CompressedImage, '/image_bev_compensated/compressed', 10)
#         self.pub_bev_white = self.create_publisher(CompressedImage, '/image_bev_white/compressed', 10)
#         self.pub_bev_yellow = self.create_publisher(CompressedImage, '/image_bev_yellow/compressed', 10)
#         self.pub_bev_roi = self.create_publisher(CompressedImage, '/image_bev_roi/compressed', 10) 

#         self.get_logger().info("DetectLane node has been initialized.")

#     def _create_perspective_transform(self):
#         """
#         src, dst 좌표는 차량의 카메라 설정에 맞게 반드시 튜닝해야 합니다.
#         원근 변환을 위한 매트릭스를 생성합니다.
#         """
#         # 이미지 크기: 1280x720 기준
#         img_size = (1280, 720)
#         # 원본 이미지의 차선 영역 4개 꼭짓점 (Source points)
#         src = np.float32([
#         (180, 400),
#         (70, 720),
#         (1230, 720),
#         (1140, 400)
#         ])
#         # src = np.float32([
#         # (460, 360),  # 왼쪽 위
#         # (340, 720),  # 왼쪽 아래
#         # (840, 720),  # 오른쪽 아래        
#         # (720, 360)  # 오른쪽 위
#         # ])

#         # 변환 후 결과 이미지의 4개 꼭짓점 (Destination points)
#         dst = np.float32([
#             (0, 0),
#             (0, img_size[1]),
#             (img_size[0], img_size[1]),
#             (img_size[0], 0)
#         ])
#         M = cv2.getPerspectiveTransform(src, dst)
#         Minv = cv2.getPerspectiveTransform(dst, src)
#         return M, Minv

#     def sync_callback(self, msg_compensated, msg_white, msg_yellow):
#         """메인 콜백 함수: 모든 처리가 여기서 시작됩니다."""

#         # =================================================================
#         # 아래 info 로그 한 줄을 추가하여 수신 성공 여부를 확인합니다.
#         # self.get_logger().info('<<< Successfully received a synchronized set of 3 messages! Starting lane detection... >>>')
#         # =================================================================
#         # self.get_logger().info(f"img_compensated shape: {img_compensated.shape}")  # 예: (720, 1280, 3)

#         # 1. 메시지 디코딩
#         # PNG 자체가 압축 포맷이고, 압축된 1채널 이미지는 encoding 정보를 무시하는 경우가 있음.
#         # 수정) mono8 명시 => 안정적으로 기본값 사용 => sliding_window 함수 실행 불가
#         # sliding_window 알고리즘은 1채널 이미지를 기준으로 만들어졌기 때문에, 3채널을 받아
#         # 차선 찾기 실패, /lane_state가 출력x.
#         img_compensated = self.cvBridge.compressed_imgmsg_to_cv2(msg_compensated)

#         # cv_bridge가 가장 잘 처리하는 기본값(bgr8)으로 먼저 디코딩
#         # 결과로 나온 3채널 이미지를 우리가 필요한 1채널 흑백 이미지로 직접 변환
#         mask_white = self.cvBridge.compressed_imgmsg_to_cv2(msg_white)
#         # 만약 3채널이라면, 1채널로 변환하여 자기 자신에게 다시 덮어씁니다.
#         if len(mask_white.shape) == 3:
#             mask_white = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
#         # 노란색 마스크도 동일하게 처리합니다.
#         mask_yellow = self.cvBridge.compressed_imgmsg_to_cv2(msg_yellow)
#         if len(mask_yellow.shape) == 3:
#             mask_yellow = cv2.cvtColor(mask_yellow, cv2.COLOR_BGR2GRAY)
#         #######
        

#         # 빨간 테두리 그리기 (원본 이미지 복사본에)
#         img_roi = img_compensated.copy()
#         pts = np.array([
#         [180, 400],  # 왼쪽 위
#         [70, 720],  # 왼쪽 아래
#         [1230, 720],  # 오른쪽 아래     
#         [1140, 400]   # 오른쪽 위
#         ], np.int32)
#         # pts = np.array([
#         # [460, 360],  # 왼쪽 위
#         # [340, 720],  # 왼쪽 아래
#         # [840, 720],  # 오른쪽 아래     
#         # [720, 360]   # 오른쪽 위
#         # ], np.int32)
#         pts = pts.reshape((-1, 1, 2))
#         cv2.polylines(img_roi, [pts], isClosed=True, color=(0, 0, 255), thickness=5)  # 빨간색( BGR = (0,0,255) ) 테두리


#         # 마스크별 픽셀 개수 출력 (디버깅용)
#         # white_pixel_count = np.count_nonzero(mask_white)
#         # yellow_pixel_count = np.count_nonzero(mask_yellow)
#         # self.get_logger().info(f"White mask pixel count: {white_pixel_count}, Yellow mask pixel count: {yellow_pixel_count}")

#         # 2. Bird's-Eye View로 원근 변환
#         warped_compensated = cv2.warpPerspective(img_compensated, self.M, (1280, 720), flags=cv2.INTER_LINEAR)
#         warped_white_mask = cv2.warpPerspective(mask_white, self.M, (1280, 720), flags=cv2.INTER_LINEAR)
#         warped_yellow_mask = cv2.warpPerspective(mask_yellow, self.M, (1280, 720), flags=cv2.INTER_LINEAR)
        
#         ## BEV 토픽 발행 설정 #######
#         bev_compensated_msg = self.cvBridge.cv2_to_compressed_imgmsg(warped_compensated)
#         bev_white_msg = self.cvBridge.cv2_to_compressed_imgmsg(warped_white_mask)
#         bev_yellow_msg = self.cvBridge.cv2_to_compressed_imgmsg(warped_yellow_mask)
#         roi_msg = self.cvBridge.cv2_to_compressed_imgmsg(img_roi)


#         self.pub_bev_compensated.publish(bev_compensated_msg)
#         self.pub_bev_white.publish(bev_white_msg)
#         self.pub_bev_yellow.publish(bev_yellow_msg)
#         self.pub_bev_roi.publish(roi_msg)
#         ##############################
#         # white_lane: left, yellow_lane: right
#         # 3. 차선 픽셀 검출 및 2차 다항식으로 피팅
#         if self.left_fit is None or self.right_fit is None:
#             # 첫 프레임이거나 차선을 놓쳤을 경우, Sliding Window로 처음부터 찾기
#             left_fitx, self.left_fit = self.sliding_window(warped_white_mask, 'left')  # white lane is left
#             right_fitx, self.right_fit = self.sliding_window(warped_yellow_mask, 'right')  # yellow lane is right
#         else:
#             # 이전 프레임의 차선 위치 주변에서 빠르게 다시 찾기
#             left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, warped_white_mask)  # white lane is left
#             right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, warped_yellow_mask)  # yellow lane is right

#         # =================================================================
#         # =================================================================
#         # [수정된 부분] 안전장치를 추가합니다.
#         # 차선 감지에 하나라도 실패했다면(None이 반환되었다면), 더 이상 진행하지 않고 현재 프레임 처리를 중단합니다.
#         if self.left_fit is None or self.right_fit is None:
#             self.get_logger().warn('Lane detection failed, skipping frame.')
            
#             # 차선을 못 찾았을 경우, 보정된 원본 이미지만이라도 발행하여 상태를 확인할 수 있습니다.
#             # (이 줄의 주석을 해제하면 됩니다.)
#             self.pub_image_lane.publish(msg_compensated)
#             return
#         # =================================================================

#         # 4. 결과 그리기 및 최종 정보 발행 (안전장치를 통과한 경우에만 실행됨)
#         self.draw_lane_and_publish(img_compensated, warped_white_mask, warped_yellow_mask, left_fitx, right_fitx)


#     def sliding_window(self, img_w, left_or_right):
#             """Sliding Window 알고리즘으로 차선을 처음부터 찾습니다."""
#             histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
#             midpoint = int(histogram.shape[0] / 2)
            
#             if left_or_right == 'left':
#                 lane_base = np.argmax(histogram[:midpoint])
#             else: # 'right'
#                 lane_base = np.argmax(histogram[midpoint:]) + midpoint

#             nwindows, margin, minpix = 20, 50, 50
#             window_height = int(img_w.shape[0] / nwindows)
#             nonzero = img_w.nonzero()
#             nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
#             x_current = lane_base
#             lane_inds = []

#             for window in range(nwindows):
#                 win_y_low = img_w.shape[0] - (window + 1) * window_height
#                 win_y_high = img_w.shape[0] - window * window_height
#                 win_x_low = x_current - margin
#                 win_x_high = x_current + margin
                
#                 good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
#                 lane_inds.append(good_inds)
                
#                 if len(good_inds) > minpix:
#                     x_current = int(np.mean(nonzerox[good_inds]))
                    
#             lane_inds = np.concatenate(lane_inds)
#             x, y = nonzerox[lane_inds], nonzeroy[lane_inds]
            
#             # [디버깅 로그 추가] 슬라이딩 윈도우를 통해 최종적으로 몇 개의 픽셀이 선택되었는지 출력합니다.
#             self.get_logger().info(f"[{left_or_right.upper()}] Pixels selected by sliding window: {len(x)}")
            
#             # 픽셀이 충분히 감지되었는지 확인
#             if len(x) < minpix * 3:  # 안정적인 피팅을 위해 최소 픽셀 수 조건 강화
#                 # 픽셀이 부족하면 이전 프레임의 값을 사용하거나, 이전 값도 없으면 None 반환
#                 previous_fit = self.left_fit if left_or_right == 'left' else self.right_fit
#                 if previous_fit is None:
#                     return None, None  # fitted_x와 lane_fit 모두 None으로 반환
#                 else:
#                     lane_fit = previous_fit
#             else:
#                 try:
#                     lane_fit = np.polyfit(y, x, 2)
#                 except TypeError:
#                     # polyfit이 실패할 경우에도 대비
#                     previous_fit = self.left_fit if left_or_right == 'left' else self.right_fit
#                     if previous_fit is None:
#                         return None, None
#                     lane_fit = previous_fit
            
#             ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
#             fitted_x = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
            
#             return fitted_x, lane_fit


#     def fit_from_lines(self, lane_fit, image):
#             """이전 차선 위치 주변에서 새로운 차선을 찾습니다."""
#             nonzero = image.nonzero()
#             nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
#             margin = 100
            
#             lane_inds = ((nonzerox > (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
#                         (nonzerox < (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin)))
            
#             x, y = nonzerox[lane_inds], nonzeroy[lane_inds]

#             try:
#                 new_fit = np.polyfit(y, x, 2)
#             except TypeError:
#                 new_fit = lane_fit

#             ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
#             fitted_x = new_fit[0] * ploty ** 2 + new_fit[1] * ploty + new_fit[2]
            
#             return fitted_x, new_fit


#     def draw_lane_and_publish(self, base_image, yellow_mask, white_mask, left_fitx, right_fitx):
#             """
#             [수정됨] 결과를 시각화하고 최종 정보를 안정적으로 발행합니다.
#             계산, 시각화, 발행의 각 단계를 명확히 분리하고 None 체크를 강화했습니다.
#             """
#             # self.get_logger().info("--- Entering draw_lane_and_publish ---")

#             # 1. 초기 설정 및 변수 준비
#             ploty = np.linspace(0, base_image.shape[0] - 1, base_image.shape[0])
#             warp_zero = np.zeros_like(yellow_mask).astype(np.uint8)
#             color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

#             # 2. 차선 상태 및 중앙선(centerx) 계산
#             # 이 단계에서는 아직 그림을 그리지 않고, 필요한 값만 계산합니다.
            
#             # 차선 검출 여부 확인
#             yellow_detected = np.count_nonzero(yellow_mask) > 3000
#             white_detected = np.count_nonzero(white_mask) > 3000

#             centerx = None  # 중앙선 좌표 배열, 계산 실패 시 None 유지

#             # [개선] 중앙선 계산 로직을 명확하게 정리   
#             # 이상적인 경우: 양쪽 차선 모두 감지
#             if left_fitx is not None and right_fitx is not None:
#                 self.get_logger().info("Calculating center from BOTH lanes.")
#                 # 두 배열의 길이가 다를 수 있으므로, 짧은 쪽에 맞춰서 안전하게 계산
#                 min_len = min(len(left_fitx), len(right_fitx))
#                 centerx = np.mean([left_fitx[:min_len], right_fitx[:min_len]], axis=0)

#             # 차선 한쪽만 감지된 경우: 감지된 차선에서 일정 거리(320px)를 오프셋하여 중앙선 추정
#             elif left_fitx is not None:
#                 self.get_logger().info("Estimating center from LEFT lane only.")
#                 centerx = left_fitx - 320
#             elif right_fitx is not None:
#                 self.get_logger().info("Estimating center from RIGHT lane only.")
#                 centerx = right_fitx + 320
#             else:
#                 # 양쪽 차선 모두 감지 실패. centerx는 그대로 None.
#                 self.get_logger().warn("Cannot calculate center, both lanes are missing.")


#             # 3. 차선 시각화 (BEV 이미지에 그리기)
#             # 이 단계에서는 계산된 값을 바탕으로 그림만 그립니다.
#             # [핵심 수정] 모든 그리기 작업 전에 None 여부를 반드시 확인합니다.

#             # 가. 왼쪽(하얀색) 차선 그리기
#             if left_fitx is not None:
#                 # ploty 배열도 left_fitx 길이에 맞춰 잘라줍니다.
#                 pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty[:len(left_fitx)]]))])
#                 cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 255, 255), thickness=25)  # white lane is left

#             # 나. 오른쪽(노란색) 차선 그리기
#             if right_fitx is not None:
#                 pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty[:len(right_fitx)]]))])
#                 cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=25)  # yellow lane is right

#             # 다. 중앙 주행 가능 영역 채우기 (양쪽 차선이 모두 있을 때만)
#             if left_fitx is not None and right_fitx is not None:
#                 min_len = min(len(left_fitx), len(right_fitx))
#                 pts_left = np.array([np.transpose(np.vstack([left_fitx[:min_len], ploty[:min_len]]))])
#                 pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[:min_len], ploty[:min_len]])))])
#                 pts = np.hstack((pts_left, pts_right))
#                 cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

#             # 라. 중앙선(추정선 포함) 그리기
#             if centerx is not None:
#                 pts_center = np.array([np.transpose(np.vstack([centerx, ploty[:len(centerx)]]))])
#                 cv2.polylines(color_warp, np.int32([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)
            

#             # 4. 최종 이미지 생성 및 정보 발행
#             # 원근 복원 및 합성
#             unwarped_lane = cv2.warpPerspective(color_warp, self.Minv, (base_image.shape[1], base_image.shape[0]))
#             final_image = cv2.addWeighted(base_image, 1, unwarped_lane, 0.5, 0)

#             # 최종 이미지 퍼블리시
#             final_image_msg = self.cvBridge.cv2_to_compressed_imgmsg(final_image, 'png')
#             self.pub_image_lane.publish(final_image_msg)

#             # 차선 상태(lane_state) 퍼블리시
#             lane_state = 0
#             if yellow_detected and white_detected: lane_state = 2
#             elif yellow_detected: lane_state = 1
#             elif white_detected: lane_state = 3
            
#             lane_state_msg = UInt8(data=lane_state)
#             self.pub_lane_state.publish(lane_state_msg)

#             # 중앙 위치 퍼블리시 (centerx가 성공적으로 계산되었을 경우에만)
#             if centerx is not None:
#                 if len(centerx) > 400:  # 로봇 앞 일정 거리의 차선 중앙값
#                     desired_center = centerx[400]
#                     self.pub_lane_center.publish(Float64(data=desired_center))
# def main(args=None):
#     rclpy.init(args=args)
#     node = DetectLane()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Keyboard Interrupt (SIGINT)')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8, Float64
from cv_bridge import CvBridge
import message_filters


def initialize_lane_fits(img_height=720, lane_center_x=640, lane_width=400):
   ploty = np.linspace(0, img_height - 1, img_height)
   left_c = lane_center_x - lane_width // 2
   right_c = lane_center_x + lane_width // 2
   left_fit = np.array([0, 0, left_c])
   right_fit = np.array([0, 0, right_c])
   left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
   right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
   return left_fit, left_fitx, right_fit, right_fitx


class DetectLane(Node):
   def __init__(self):
       super().__init__('detect_lane')
       self.cvBridge = CvBridge()
       self.M, self.Minv = self._create_perspective_transform()
       self.left_fit, self.left_fitx, self.right_fit, self.right_fitx = initialize_lane_fits()


       # Subscriber (message_filters)
       sub_compensated = message_filters.Subscriber(self, CompressedImage, '/image_compensated/compressed')
       sub_white_mask = message_filters.Subscriber(self, CompressedImage, '/image_compensated/white_mask/compressed')
       sub_yellow_mask = message_filters.Subscriber(self, CompressedImage, '/image_compensated/yellow_mask/compressed')
       self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
           [sub_compensated, sub_white_mask, sub_yellow_mask],
           queue_size=10,
           slop=0.1
       )
       self.time_synchronizer.registerCallback(self.sync_callback)


       # Publisher
       self.pub_image_lane = self.create_publisher(CompressedImage, '/image_lane_detected/compressed', 10)
       self.pub_lane_state = self.create_publisher(UInt8, '/lane_state', 10)
       self.pub_lane_center = self.create_publisher(Float64, '/lane_center', 10)
       self.pub_bev_compensated = self.create_publisher(CompressedImage, '/image_bev_compensated/compressed', 10)
       self.pub_bev_white = self.create_publisher(CompressedImage, '/image_bev_white/compressed', 10)
       self.pub_bev_yellow = self.create_publisher(CompressedImage, '/image_bev_yellow/compressed', 10)
       self.pub_bev_roi = self.create_publisher(CompressedImage, '/image_bev_roi/compressed', 10)


       self.get_logger().info("DetectLane node has been initialized.")


   def _create_perspective_transform(self):
       img_size = (1280, 720)
       src = np.float32([
        (20, 400),
        (0, 720),
        (1280, 720),
        (1260, 400)
       ])
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
       img_compensated = self.cvBridge.compressed_imgmsg_to_cv2(msg_compensated)
       mask_white = self.cvBridge.compressed_imgmsg_to_cv2(msg_white)
       mask_yellow = self.cvBridge.compressed_imgmsg_to_cv2(msg_yellow)
       if len(mask_white.shape) == 3:
           mask_white = cv2.cvtColor(mask_white, cv2.COLOR_BGR2GRAY)
       if len(mask_yellow.shape) == 3:
           mask_yellow = cv2.cvtColor(mask_yellow, cv2.COLOR_BGR2GRAY)


       # ROI 시각화
       img_roi = img_compensated.copy()
       pts = np.array([[20, 400], [0, 720], [1280, 720], [1260, 400]], np.int32).reshape((-1, 1, 2))
       cv2.polylines(img_roi, [pts], isClosed=True, color=(0, 0, 255), thickness=5)


       # BEV 변환
       warped_compensated = cv2.warpPerspective(img_compensated, self.M, (1280, 720))
       warped_white = cv2.warpPerspective(mask_white, self.M, (1280, 720))
       warped_yellow = cv2.warpPerspective(mask_yellow, self.M, (1280, 720))


       # BEV 이미지 발행
       self.pub_bev_compensated.publish(self.cvBridge.cv2_to_compressed_imgmsg(warped_compensated))
       self.pub_bev_white.publish(self.cvBridge.cv2_to_compressed_imgmsg(warped_white))
       self.pub_bev_yellow.publish(self.cvBridge.cv2_to_compressed_imgmsg(warped_yellow))
       self.pub_bev_roi.publish(self.cvBridge.cv2_to_compressed_imgmsg(img_roi))


       # 차선 fitting
       self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, warped_yellow)
       self.right_fitx, self.right_fit = self.fit_from_lines(self.right_fit, warped_white)


       # 시각화 및 최종 발행
       self.draw_lane_and_publish(img_compensated, warped_yellow, warped_white, self.left_fitx, self.right_fitx)


   def fit_from_lines(self, lane_fit, image):
       nonzero = image.nonzero()
       nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
       margin = 300
       lane_inds = ((nonzerox > (lane_fit[0] * nonzeroy**2 + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
                    (nonzerox < (lane_fit[0] * nonzeroy**2 + lane_fit[1] * nonzeroy + lane_fit[2] + margin)))
       x, y = nonzerox[lane_inds], nonzeroy[lane_inds]


       if len(x) == 0:
           ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
           fitx = lane_fit[0] * ploty**2 + lane_fit[1] * ploty + lane_fit[2]
           self.get_logger().warn("No pixels found near previous lane_fit. Using previous fit.")
           return fitx, lane_fit


       try:
           new_fit = np.polyfit(y, x, 2)
       except Exception as e:
           self.get_logger().warn(f"Polyfit failed: {e}")
           new_fit = lane_fit


       ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
       fitx = new_fit[0] * ploty**2 + new_fit[1] * ploty + new_fit[2]
       return fitx, new_fit


   def draw_lane_and_publish(self, base_image, yellow_mask, white_mask, left_fitx, right_fitx):
       ploty = np.linspace(0, base_image.shape[0] - 1, base_image.shape[0])
       warp_zero = np.zeros_like(yellow_mask).astype(np.uint8)
       color_warp = np.dstack((warp_zero, warp_zero, warp_zero))


       yellow_detected = np.count_nonzero(yellow_mask) > 3000
       white_detected = np.count_nonzero(white_mask) > 3000


       lane_state = 0
       if yellow_detected and white_detected:
           lane_state = 2
       elif yellow_detected:
           lane_state = 1
       elif white_detected:
           lane_state = 3


       centerx = None
       if lane_state == 2:
           centerx = np.mean([left_fitx, right_fitx], axis=0)
           pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
           pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
           pts = np.hstack((pts_left, pts_right))
           cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
       elif lane_state == 1:
           centerx = left_fitx + 320
       elif lane_state == 3:
           centerx = right_fitx - 320


       if yellow_detected:
           pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
           cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 255, 0), thickness=25)
       if white_detected:
           pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
           cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)
       if centerx is not None:
           pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
           cv2.polylines(color_warp, np.int32([pts_center]), isClosed=False, color=(0, 255, 255), thickness=12)


       unwarped = cv2.warpPerspective(color_warp, self.Minv, (base_image.shape[1], base_image.shape[0]))
       final = cv2.addWeighted(base_image, 1, unwarped, 0.5, 0)


       self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'png'))
       self.pub_lane_state.publish(UInt8(data=lane_state))
       if centerx is not None:
           self.pub_lane_center.publish(Float64(data=centerx[400]))


def main(args=None):
   rclpy.init(args=args)
   node = DetectLane()
   try:
       rclpy.spin(node)
   except KeyboardInterrupt:
       pass
   finally:
       node.destroy_node()
       rclpy.shutdown()


if __name__ == '__main__':
   main()





