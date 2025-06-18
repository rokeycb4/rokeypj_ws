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
