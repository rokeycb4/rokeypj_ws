import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.sub_image_type = 'compressed'
        self.pub_image_type = 'compressed'

        # subscribe_topic = '/camera/image_raw/compressed'
        # self.sub_image_original = self.create_subscription(
        #     CompressedImage, subscribe_topic, self.cbFindLane, 1
        # )

        subscribe_topic = '/camera/preprocessed/compressed'
        self.sub_image_original = self.create_subscription(
            CompressedImage, subscribe_topic, self.cbFindLane, 1
        )


        self.pub_image_lane = self.create_publisher(
            CompressedImage, '/detect/image_output/compressed', 1
        )
        self.pub_image_yellow_lane = self.create_publisher(
            CompressedImage, '/detect/image_output_yellow/compressed', 1
        )
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
        )
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()
        self.counter = 1

        self.window_width = 1000.
        self.window_height = 600.

        self.reliability_yellow_line = 100
        self.mov_avg_left = np.empty((0, 3))

        initial_img_width = 1280
        initial_img_height = 720
        ploty_init = np.linspace(0, initial_img_height - 1, initial_img_height)

        self.left_fit = np.array([0.0, 0.0, initial_img_width / 2 - 150])
        self.left_fitx = self.left_fit[0] * ploty_init**2 + self.left_fit[1] * ploty_init + self.left_fit[2]
        self.lane_fit_bef = np.array([0., 0., 0.])

        self.hsv_yellow_lower = [10, 50, 60]
        self.hsv_yellow_upper = [55, 255, 255]

        self.yellow_offset = 300

    def cbFindLane(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])
        yellow_fraction, cv_yellow_lane = self.maskYellowLane(cv_image)

        try:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.fit_from_lines(self.left_fit, cv_yellow_lane)
                self.mov_avg_left = np.append(self.mov_avg_left, np.array([self.left_fit]), axis=0)
        except Exception:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.sliding_windown(cv_yellow_lane, 'left')
                self.mov_avg_left = np.array([self.left_fit])

        MOV_AVG_LENGTH = 5
        if self.mov_avg_left.shape[0] > 0:
            self.left_fit = np.array([
                np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])
        else:
            self.get_logger().warn('No left lane data yet for moving average. Using default left_fit.')

        self.left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]

        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]

        self.make_lane(cv_image, yellow_fraction)

    def maskYellowLane(self, image):
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array(self.hsv_yellow_lower)
        upper_yellow = np.array(self.hsv_yellow_upper)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fraction_num = np.count_nonzero(mask)

        how_much_short = 600 - np.count_nonzero(np.count_nonzero(mask, axis=1))
        if how_much_short > 100:
            self.reliability_yellow_line = max(0, self.reliability_yellow_line - 5)
        else:
            self.reliability_yellow_line = min(100, self.reliability_yellow_line + 5)

        msg = UInt8()
        msg.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg)

        # ⬇️ yellow mask 이미지 발행
        self.pub_image_yellow_lane.publish(
            self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
        )

        return fraction_num, mask

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_inds = (
            (nonzerox > (lane_fit[0] * (nonzeroy**2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
            (nonzerox < (lane_fit[0] * (nonzeroy**2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin))
        )

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        lane_fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty**2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def sliding_windown(self, img_w, side):
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)

        lane_base = np.argmax(histogram[:midpoint]) if side == 'left' else np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20
        window_height = int(img_w.shape[0] / nwindows)

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

            good_lane_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) & (nonzerox < win_x_high)
            ).nonzero()[0]

            lane_inds.append(good_lane_inds)

            if len(good_lane_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_lane_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except Exception:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty**2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def make_lane(self, cv_image, yellow_fraction):
        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])
        lane_state = UInt8()
        centerx = np.array([cv_image.shape[1] / 2] * cv_image.shape[0])
        self.is_center_x_exist = False

        if yellow_fraction > 3000:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            cv2.polylines(cv_image, np.int32([pts_left]), False, (0, 255, 255), 12)

            centerx = np.add(self.left_fitx, self.yellow_offset)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            cv2.polylines(cv_image, np.int32([pts_center]), False, (255, 255, 255), 8)

            # 텍스트 추가
            cv2.putText(cv_image, "yellow lane", (int(self.left_fitx[-1]) - 100, int(ploty[-1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(cv_image, "center", (int(centerx[-1]) + 10, int(ploty[-1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 5)

            self.is_center_x_exist = True
            lane_state.data = 1

        self.pub_lane_state.publish(lane_state)
        self.get_logger().info(f'Lane state: {lane_state.data}')
        if self.is_center_x_exist:
            self.get_logger().info(f'Desired center X: {centerx.item(350)}')

        if self.pub_image_type == 'compressed':
            if self.is_center_x_exist:
                msg = Float64()
                msg.data = centerx.item(350)
                self.pub_lane.publish(msg)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(cv_image, 'jpg'))
        else:
            if self.is_center_x_exist:
                msg = Float64()
                msg.data = centerx.item(350)
                self.pub_lane.publish(msg)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(cv_image, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
