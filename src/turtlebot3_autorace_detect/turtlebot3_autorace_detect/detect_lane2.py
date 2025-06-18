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
