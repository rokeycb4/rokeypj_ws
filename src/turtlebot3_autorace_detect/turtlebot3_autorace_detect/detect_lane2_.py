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
