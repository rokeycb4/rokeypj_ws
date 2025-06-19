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

        self.sub_image_original = self.create_subscription(
            CompressedImage, '/camera/preprocessed/compressed',
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
        self.pub_trapezoid_region = self.create_publisher(
            CompressedImage, '/detect/trapezoid_region/compressed', 1
        )

        self.cvBridge = CvBridge()
        self.counter = 1

        self.reliability_yellow_line = 100
        self.mov_avg_left = np.empty((0, 3))

        self.yellow_offset = 550

        # 노란색 HSV 범위 (확장)
        self.hsv_yellow_lower = [4, 15, 100]
        self.hsv_yellow_upper = [65, 255, 255]

        # 흰색 HSV 범위 (정의만, 사용 안 함)
        self.hsv_white_lower = [0, 0, 200]
        self.hsv_white_upper = [180, 45, 255]

        src = np.float32([
            [0, 300],
            [0, 720],           # left down
            [1230, 720],
            [1140, 400]
        ])
        dst = np.float32([
            [0, 0],
            [0, 720],
            [1280, 720],
            [1280, 0]
        ])
        self.Msrc = src
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

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.pub_image_type == 'compressed':
            self.pub_image_mask.publish(self.cvBridge.cv2_to_compressed_imgmsg(mask_bgr, 'jpg'))

        # 원본 이미지 위에 사다리꼴 영역 시각화
        visualized = raw_image.copy()
        pts = np.array(self.Msrc, np.int32).reshape((-1, 1, 2))
        cv2.polylines(visualized, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        trap_msg = self.cvBridge.cv2_to_compressed_imgmsg(visualized, 'jpg')
        self.pub_trapezoid_region.publish(trap_msg)

    def maskYellowLane(self, image):
        height, width = image.shape[:2]
        roi = image[:, :int(width * 0.6)]

        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, np.array(self.hsv_yellow_lower), np.array(self.hsv_yellow_upper))
        mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

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
            cx_val = float(centerx.item(550))
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
                    cv2.circle(color_image, (lx, y), 2, (0, 255, 255), -1)

            if self.is_center_x_exist:
                centerx_int = (self.left_fitx + self.yellow_offset).astype(np.int32)
                for y, cx in enumerate(centerx_int):
                    if 0 <= y < image.shape[0] and 0 <= cx < image.shape[1]:
                        cv2.circle(color_image, (cx, y), 2, (0, 0, 255), -1)

                y_text = 550
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
