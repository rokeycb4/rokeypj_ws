#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange, ParameterDescriptor, SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64, UInt8

class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        hue_desc = ParameterDescriptor(description='hue', integer_range=[IntegerRange(from_value=0, to_value=179, step=1)])
        sl_desc = ParameterDescriptor(description='saturation_lightness', integer_range=[IntegerRange(from_value=0, to_value=255, step=1)])

        self.declare_parameters(
            '',
            [
                ('detect.lane.white.hue_l', 0, hue_desc),
                ('detect.lane.white.hue_h', 179, hue_desc),
                ('detect.lane.white.saturation_l', 0, sl_desc),
                ('detect.lane.white.saturation_h', 70, sl_desc),
                ('detect.lane.white.lightness_l', 105, sl_desc),
                ('detect.lane.white.lightness_h', 255, sl_desc),
                ('detect.lane.yellow.hue_l', 10, hue_desc),
                ('detect.lane.yellow.hue_h', 127, hue_desc),
                ('detect.lane.yellow.saturation_l', 70, sl_desc),
                ('detect.lane.yellow.saturation_h', 255, sl_desc),
                ('detect.lane.yellow.lightness_l', 95, sl_desc),
                ('detect.lane.yellow.lightness_h', 255, sl_desc),
                ('is_detection_calibration_mode', False)
            ]
        )

        self.load_parameters()

        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.param_callback)

        # 구독
        self.sub_image_original = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.cbFindLaneCompressed, 1)
        self.sub_image_preprocessed = self.create_subscription(
            Image, '/camera/image', self.cbFindLaneRawPreprocessed, 1)
        self.sub_image_compensated = self.create_subscription(
            Image, '/camera/image_compensated', self.cbFindLaneRawCompensated, 1)

        # 퍼블리셔 (이 부분만 토픽명 변경)
        self.pub_image_lane = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_image_lane_raw = self.create_publisher(CompressedImage, '/detect/image_output/org/compressed', 1)
        self.pub_image_lane_preprocessed = self.create_publisher(CompressedImage, '/detect/image_output/preprocessed/compressed', 1)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        if self.is_calibration_mode:
            self.pub_image_white_lane = self.create_publisher(CompressedImage, '/detect/white_lane_mask/compressed', 1)
            self.pub_image_yellow_lane = self.create_publisher(CompressedImage, '/detect/yellow_lane_mask/compressed', 1)

        self.cvBridge = CvBridge()
        self.counter_compressed = 1
        self.counter_preprocessed = 1
        self.counter_compensated = 1

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

        self.is_calibration_mode = p('is_detection_calibration_mode').get_parameter_value().bool_value

    def param_callback(self, parameters):
        for param in parameters:
            if param.name.startswith('detect.lane'):
                self.set_parameters([param])
        self.load_parameters()
        return SetParametersResult(successful=True)

    # compressed 구독
    def cbFindLaneCompressed(self, image_msg):
        if self.counter_compressed % 3 != 0:
            self.counter_compressed += 1
            return
        self.counter_compressed = 1
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.process_lane_visualize(cv_image, self.pub_image_lane_raw)

    # preprocessed 구독
    def cbFindLaneRawPreprocessed(self, image_msg):
        if self.counter_preprocessed % 3 != 0:
            self.counter_preprocessed += 1
            return
        self.counter_preprocessed = 1
        cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
        self.process_lane_visualize(cv_image, self.pub_image_lane_preprocessed)

    # compensated 구독 (full 검출)
    def cbFindLaneRawCompensated(self, image_msg):
        if self.counter_compensated % 3 != 0:
            self.counter_compensated += 1
            return
        self.counter_compensated = 1
        cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
        self.process_lane_full(cv_image)

    # full 퍼블리시용 검출
    def process_lane_full(self, cv_image):
        white_mask = self.mask_lane(cv_image, 'white')
        yellow_mask = self.mask_lane(cv_image, 'yellow')

        if self.is_calibration_mode:
            self.pub_image_white_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(white_mask, 'jpg'))
            self.pub_image_yellow_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(yellow_mask, 'jpg'))

        final, state, center_x = self.make_lane(cv_image, white_mask, yellow_mask)
        self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))
        self.pub_lane_state.publish(UInt8(data=state))
        if center_x is not None:
            self.pub_lane.publish(Float64(data=float(center_x)))
            self.get_logger().info(f"CenterX: {center_x}")

    # 시각화 전용 검출
    def process_lane_visualize(self, cv_image, publisher):
        white_mask = self.mask_lane(cv_image, 'white')
        yellow_mask = self.mask_lane(cv_image, 'yellow')
        final, _, _ = self.make_lane(cv_image, white_mask, yellow_mask)
        publisher.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

    def mask_lane(self, image, color):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if color == 'white':
            lower = np.array([self.hue_white_l, self.saturation_white_l, self.lightness_white_l])
            upper = np.array([self.hue_white_h, self.saturation_white_h, self.lightness_white_h])
        else:
            lower = np.array([self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l])
            upper = np.array([self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h])
        return cv2.inRange(hsv, lower, upper)

    def sliding_window(self, mask, side):
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
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

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                         & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x, y = nonzerox[lane_inds], nonzeroy[lane_inds]
        fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        return fitx, ploty

    def make_lane(self, cv_image, white_mask, yellow_mask):
        yellow_exist = np.count_nonzero(yellow_mask) > 3000
        white_exist = np.count_nonzero(white_mask) > 3000
        warp = np.zeros_like(cv_image)
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

        final = cv2.addWeighted(cv_image, 1, warp, 1, 0)
        return final, state, center_x

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
