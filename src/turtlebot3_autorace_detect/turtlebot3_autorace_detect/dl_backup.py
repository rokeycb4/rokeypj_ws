

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
