#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
#   - Special Thanks : Roger Sacchelli

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_msgs.msg import UInt8


class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=179,
                step=1)]
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            description='saturation and lightness range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=255,
                step=1)]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detect.lane.white.hue_l', 0,
                    parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179,
                    parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 10,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.saturation_h', 60,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_l', 180,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.hue_l', 20,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 40,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 50,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.saturation_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_l', 60,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('is_detection_calibration_mode', False)
            ]
        )

        self.hue_white_l = self.get_parameter(
            'detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter(
            'detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter(
            'detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter(
            'detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = self.get_parameter(
            'detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = self.get_parameter(
            'detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'detect.lane.yellow.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_type = 'compressed' #rujin raw->compressed         # you can choose image type 'compressed', 'raw'
        self.pub_image_type = 'compressed'  # you can choose image type 'compressed', 'raw'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                #CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                CompressedImage, '/image_raw/compressed', self.cbFindLane, 1                
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1
                )
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1
                )

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1
                    )
            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1
                    )

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)

        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
            )

        self.pub_white_line_reliability = self.create_publisher(
            UInt8, '/detect/white_line_reliability', 1
            )

        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()

        self.counter = 1

        self.window_width = 1000.
        self.window_height = 600.

        self.reliability_white_line = 100
        self.reliability_yellow_line = 100

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))

        # Add these initializations

        initial_img_height = 720 
        initial_img_width = 1280  
        ploty_init = np.linspace(0, initial_img_height - 1, initial_img_height)

        self.left_fit = np.array([0.0, 0.0, initial_img_width / 2 - 150]) 
        self.right_fit = np.array([0.0, 0.0, initial_img_width / 2 + 150])
        self.lane_fit_bef = np.array([0., 0., 0.])

        self.left_fitx = self.left_fit[0] * ploty_init**2 + self.left_fit[1] * ploty_init + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * ploty_init**2 + self.right_fit[1] * ploty_init + self.right_fit[2]
        self.lane_fit_bef = np.array([0., 0., 0.])

        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 50
        self.prev_centerx = None
        self.alpha = 0.2
        self.roi_vertices = np.array([
            [(100, 600), (400, 350), (880, 350), (1180, 600)]
        ], dtype=np.int32)

    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            self.get_logger().info(f'Parameter name: {param.name}')
            self.get_logger().info(f'Parameter value: {param.value}')
            self.get_logger().info(f'Parameter type: {param.type_}')
            if param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.white.hue_h':
                self.hue_white_h = param.value
            elif param.name == 'detect.lane.white.saturation_l':
                self.saturation_white_l = param.value
            elif param.name == 'detect.lane.white.saturation_h':
                self.saturation_white_h = param.value
            elif param.name == 'detect.lane.white.lightness_l':
                self.lightness_white_l = param.value
            elif param.name == 'detect.lane.white.lightness_h':
                self.lightness_white_h = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.hue_h':
                self.hue_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.saturation_l':
                self.saturation_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.saturation_h':
                self.saturation_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.lightness_l':
                self.lightness_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.lightness_h':
                self.lightness_yellow_h = param.value
            return SetParametersResult(successful=True)

    # def cbFindLane(self, image_msg):
    #     # Change the frame rate by yourself. Now, it is set to 1/3 (10fps).
    #     # Unappropriate value of frame rate may cause huge delay on entire recognition process.
    #     # This is up to your computer's operating power.
    #     if self.counter % 3 != 0:
    #         self.counter += 1
    #         return
    #     else:
    #         self.counter = 1

    #     if self.sub_image_type == 'compressed':
    #         np_arr = np.frombuffer(image_msg.data, np.uint8)
    #         cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #     elif self.sub_image_type == 'raw':
    #         cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

    #     # ploty defination
    #     ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

    #     white_fraction, cv_white_lane = self.maskWhiteLane(cv_image)
    #     yellow_fraction, cv_yellow_lane = self.maskYellowLane(cv_image)

    #     try:
    #         if yellow_fraction > 3000:
    #             self.left_fitx, self.left_fit = self.fit_from_lines(
    #                 self.left_fit, cv_yellow_lane)
    #             self.mov_avg_left = np.append(
    #                 self.mov_avg_left, np.array([self.left_fit]), axis=0
    #                 )

    #         if white_fraction > 3000:
    #             self.right_fitx, self.right_fit = self.fit_from_lines(
    #                 self.right_fit, cv_white_lane)
    #             self.mov_avg_right = np.append(
    #                 self.mov_avg_right, np.array([self.right_fit]), axis=0
    #                 )
    #     except Exception:
    #         if yellow_fraction > 3000:
    #             self.left_fitx, self.left_fit = self.sliding_windown(cv_yellow_lane, 'left')
    #             self.mov_avg_left = np.array([self.left_fit])

    #         if white_fraction > 3000:
    #             self.right_fitx, self.right_fit = self.sliding_windown(cv_white_lane, 'right')
    #             self.mov_avg_right = np.array([self.right_fit])

    #     MOV_AVG_LENGTH = 5
    #     if self.mov_avg_left.shape[0] > 0:  # mov_avg_left 배열에 데이터가 있을 때만 평균을 계산
    #         self.left_fit = np.array([
    #             np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
    #             np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
    #             np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])
    #             ])
    #     else:
    #         # 데이터가 없으면 __init__에서 설정한 기본 left_fit 값을 유지
    #         self.get_logger().warn('No left lane data yet for moving average. Using default left_fit.')

    #     # left_fit이 업데이트되었든 기본값이든, 항상 left_fitx를 새로 계산
    #     self.left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]

    #     # mov_avg_right 배열에 데이터가 있을 때만 평균을 계산
    #     if self.mov_avg_right.shape[0] > 0:
    #         self.right_fit = np.array([
    #             np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
    #             np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
    #             np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])
    #             ])
    #     else:
    #         # 데이터가 없으면 __init__에서 설정한 기본 right_fit 값을 유지
    #         self.get_logger().warn('No right lane data yet for moving average. Using default right_fit.')

    #     # right_fit이 업데이트되었든 기본값이든, 항상 right_fitx를 새로 계산
    #     self.right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

    #     if self.mov_avg_left.shape[0] > 1000:
    #         self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]

    #     if self.mov_avg_right.shape[0] > 1000:
    #         self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]

    #     self.make_lane(cv_image, white_fraction, yellow_fraction)

    # def maskWhiteLane(self, image):

    #     image = cv2.GaussianBlur(image, (5,5), 0)   # gaussian blur
    #     image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)  # color contrast

    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #     Hue_l = self.hue_white_l
    #     Hue_h = self.hue_white_h
    #     Saturation_l = self.saturation_white_l
    #     Saturation_h = self.saturation_white_h
    #     Lightness_l = self.lightness_white_l
    #     Lightness_h = self.lightness_white_h

    #     lower_white = np.array([Hue_l, Saturation_l, Lightness_l])
    #     upper_white = np.array([Hue_h, Saturation_h, Lightness_h])

    #     mask = cv2.inRange(hsv, lower_white, upper_white)

    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # mophology calculation

    #     fraction_num = np.count_nonzero(mask)

    #     if not self.is_calibration_mode:
    #         if fraction_num > 35000:
    #             if self.lightness_white_l < 250:
    #                 self.lightness_white_l += 5
    #         elif fraction_num < 5000:
    #             if self.lightness_white_l > 50:
    #                 self.lightness_white_l -= 5

    #     how_much_short = 0

    #     for i in range(0, 600):
    #         if np.count_nonzero(mask[i, ::]) > 0:
    #             how_much_short += 1

    #     how_much_short = 600 - how_much_short

    #     if how_much_short > 100:
    #         if self.reliability_white_line >= 5:
    #             self.reliability_white_line -= 5
    #     elif how_much_short <= 100:
    #         if self.reliability_white_line <= 99:
    #             self.reliability_white_line += 5

    #     msg_white_line_reliability = UInt8()
    #     msg_white_line_reliability.data = self.reliability_white_line
    #     self.pub_white_line_reliability.publish(msg_white_line_reliability)

    #     if self.is_calibration_mode:
    #         if self.pub_image_type == 'compressed':
    #             self.pub_image_white_lane.publish(
    #                 self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
    #                 )

    #         elif self.pub_image_type == 'raw':
    #             self.pub_image_white_lane.publish(
    #                 self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
    #                 )

    #     return fraction_num, mask

    # def maskYellowLane(self, image):
        
    #     image = cv2.GaussianBlur(image, (5,5), 0)   # gaussian blur
    #     image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)  # color contrast

    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #     Hue_l = self.hue_yellow_l
    #     Hue_h = self.hue_yellow_h
    #     Saturation_l = self.saturation_yellow_l
    #     Saturation_h = self.saturation_yellow_h
    #     Lightness_l = self.lightness_yellow_l
    #     Lightness_h = self.lightness_yellow_h

    #     lower_yellow = np.array([Hue_l, Saturation_l, Lightness_l])
    #     upper_yellow = np.array([Hue_h, Saturation_h, Lightness_h])

    #     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # mophology calculation

    #     fraction_num = np.count_nonzero(mask)

    #     if self.is_calibration_mode:
    #         if fraction_num > 35000:
    #             if self.lightness_yellow_l < 250:
    #                 self.lightness_yellow_l += 20
    #         elif fraction_num < 5000:
    #             if self.lightness_yellow_l > 90:
    #                 self.lightness_yellow_l -= 20

    #     how_much_short = 0

    #     for i in range(0, 600):
    #         if np.count_nonzero(mask[i, ::]) > 0:
    #             how_much_short += 1

    #     how_much_short = 600 - how_much_short

    #     if how_much_short > 100:
    #         if self.reliability_yellow_line >= 5:
    #             self.reliability_yellow_line -= 5
    #     elif how_much_short <= 100:
    #         if self.reliability_yellow_line <= 99:
    #             self.reliability_yellow_line += 5

    #     msg_yellow_line_reliability = UInt8()
    #     msg_yellow_line_reliability.data = self.reliability_yellow_line
    #     self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

    #     if self.is_calibration_mode:
    #         if self.pub_image_type == 'compressed':
    #             self.pub_image_yellow_lane.publish(
    #                 self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
    #                 )

    #         elif self.pub_image_type == 'raw':
    #             self.pub_image_yellow_lane.publish(
    #                 self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
    #                 )

    #     return fraction_num, mask

    # def fit_from_lines(self, lane_fit, image):
    #     nonzero = image.nonzero()
    #     nonzeroy = np.array(nonzero[0])
    #     nonzerox = np.array(nonzero[1])
    #     margin = 100
    #     lane_inds = (
    #         (nonzerox >
    #             (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
    #         (nonzerox <
    #             (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin))
    #             )

    #     x = nonzerox[lane_inds]
    #     y = nonzeroy[lane_inds]

    #     lane_fit = np.polyfit(y, x, 2)

    #     ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    #     lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

    #     return lane_fitx, lane_fit

    # def make_lane(self, cv_image, white_fraction, yellow_fraction):
    #     # Create an image to draw the lines on
    #     warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)

    #     color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #     color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

    #     ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

    #     # both lane -> 2, left lane -> 1, right lane -> 3, none -> 0
    #     lane_state = UInt8()

    #     # 함수 시작 시 centerx를 기본값(이미지 중앙)으로 초기화
    #     centerx = np.array([cv_image.shape[1] / 2] * cv_image.shape[0]) # 기본값으로 이미지 중앙 X좌표 사용

    #     # 유효한 차선 중앙이 계산될 때만 True
    #     self.is_center_x_exist = False

    #     if yellow_fraction > 3000:
    #         pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
    #         cv2.polylines(
    #             color_warp_lines,
    #             np.int_([pts_left]),
    #             isClosed=False,
    #             color=(0, 0, 255),
    #             thickness=25
    #             )

    #     if white_fraction > 3000:
    #         pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
    #         cv2.polylines(
    #             color_warp_lines,
    #             np.int_([pts_right]),
    #             isClosed=False,
    #             color=(255, 255, 0),
    #             thickness=25
    #             )

    #     if self.reliability_white_line > 50 and self.reliability_yellow_line > 50:
    #         if white_fraction > 3000 and yellow_fraction > 3000:
    #             centerx = np.mean([self.left_fitx, self.right_fitx], axis=0)
    #             pts = np.hstack((pts_left, pts_right))
    #             pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

    #             cv2.polylines(
    #                 color_warp_lines,
    #                 np.int_([pts_center]),
    #                 isClosed=False,
    #                 color=(0, 255, 255),
    #                 thickness=12
    #                 )

    #             # Draw the lane onto the warped blank image
    #             cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    #             lane_state.data = 2
    #             self.is_center_x_exist = True   # 이 경우 centerx가 유효하게 계산

    #         elif white_fraction > 3000 and yellow_fraction <= 3000:
    #             centerx = np.subtract(self.right_fitx, 280)
    #             pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

    #             cv2.polylines(
    #                 color_warp_lines,
    #                 np.int_([pts_center]),
    #                 isClosed=False,
    #                 color=(0, 255, 255),
    #                 thickness=12
    #                 )
                
    #             lane_state.data = 3
    #             self.is_center_x_exist = True # 이 경우 centerx가 유효하게 계산

    #         elif white_fraction <= 3000 and yellow_fraction > 3000:
    #             centerx = np.add(self.left_fitx, 280)
    #             pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

    #             cv2.polylines(
    #                 color_warp_lines,
    #                 np.int_([pts_center]),
    #                 isClosed=False,
    #                 color=(0, 255, 255),
    #                 thickness=12
    #                 )
                
    #             lane_state.data = 1
    #             self.is_center_x_exist = True # 이 경우 centerx가 유효하게 계산
                
    #         else: # reliability는 높지만, white_fraction/yellow_fraction이 충분치 않은 경우
    #             lane_state.data = 0
    #             self.get_logger().warn('High reliability, but insufficient lane fraction detected. Lane state: 0')

    #     elif self.reliability_white_line <= 50 and self.reliability_yellow_line > 50:
    #         centerx = np.add(self.left_fitx, 280)
    #         pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

    #         cv2.polylines(
    #             color_warp_lines,
    #             np.int_([pts_center]),
    #             isClosed=False,
    #             color=(0, 255, 255),
    #             thickness=12
    #             )
            
    #         lane_state.data = 1
    #         self.is_center_x_exist = True # 이 경우 centerx가 유효하게 계산

    #     elif self.reliability_white_line > 50 and self.reliability_yellow_line <= 50:
    #         centerx = np.subtract(self.right_fitx, 280)
    #         pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

    #         cv2.polylines(
    #             color_warp_lines,
    #             np.int_([pts_center]),
    #             isClosed=False,
    #             color=(0, 255, 255),
    #             thickness=12
    #             )
            
    #         lane_state.data = 3
    #         self.is_center_x_exist = True # 이 경우 centerx가 유효하게 계산

    #     else:   # 어떤 유효한 차선 감지 조건도 만족하지 못하는 경우
    #         self.is_center_x_exist = False  # centerx가 유효하지 않음을 명확히 표시
    #         lane_state.data = 0
    #         self.get_logger().info('Low reliability for both white and yellow lines. Lane state: 0')
    #         # 이 경우 centerx는 함수 시작 시 초기화된 기본값을 유지

    #         pass

    #     self.pub_lane_state.publish(lane_state)
    #     self.get_logger().info(f'Lane state: {lane_state.data}')

    #     # Combine the result with the original image
    #     final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
    #     final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

    #     if self.pub_image_type == 'compressed':
    #         if self.is_center_x_exist:
    #             # publishes lane center
    #             msg_desired_center = Float64()
    #             msg_desired_center.data = centerx.item(350)
    #             self.pub_lane.publish(msg_desired_center)

    #         self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))
        
    #     elif self.pub_image_type == 'raw':
    #         if self.is_center_x_exist:
    #             # publishes lane center
    #             msg_desired_center = Float64()
    #             msg_desired_center.data = centerx.item(350)
    #             self.pub_lane.publish(msg_desired_center)

    #         self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(final, 'bgr8'))

    def preprocess_image(self, image):
        """이미지 전처리 함수"""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

    def create_roi_mask(self, image):
        """ROI 마스크 생성"""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image):
        """에지 검출"""
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges

    def detect_lines(self, edges):
        """허프 변환을 통한 직선 검출"""
        return cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=self.hough_threshold,
                            minLineLength=self.min_line_length,
                            maxLineGap=self.max_line_gap)

    def separate_lines(self, lines):
        """검출된 선을 좌/우 차선으로 분리"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.3:  # 왼쪽 차선
                left_lines.append(line)
            elif slope > 0.3:  # 오른쪽 차선
                right_lines.append(line)
                
        return left_lines, right_lines
    
    def fit_from_lines(self, lines, image_shape):
        """차선 피팅"""
        if not lines:
            return None, None
            
        points = np.concatenate([line[0] for line in lines])
        x = points[:, [0, 2]].flatten()
        y = points[:, [1, 3]].flatten()
        
        try:
            coefficients = np.polyfit(y, x, 2)
            ploty = np.linspace(0, image_shape[0] - 1, image_shape[0])
            fitx = coefficients[0] * ploty ** 2 + coefficients[1] * ploty + coefficients[2]
            
            return fitx, coefficients
        except:
            return None, None

    def make_lane(self, cv_image, left_fit, right_fit):
        """차선 시각화"""
        warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
        
        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])
        lane_state = UInt8()
        centerx = np.array([cv_image.shape[1] / 2] * cv_image.shape[0])
        
        if left_fit is not None:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fit, ploty])))])
            cv2.polylines(color_warp_lines, np.int_([pts_left]), False, (0, 0, 255), 25)
            
        if right_fit is not None:
            pts_right = np.array([np.transpose(np.vstack([right_fit, ploty]))])
            cv2.polylines(color_warp_lines, np.int_([pts_right]), False, (255, 255, 0), 25)
            
        if left_fit is not None and right_fit is not None:
            centerx = np.mean([left_fit, right_fit], axis=0)
            pts = np.hstack((pts_left, pts_right))
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])
            
            cv2.polylines(color_warp_lines, np.int_([pts_center]), False, (0, 255, 255), 12)
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            
            lane_state.data = 2
        elif right_fit is not None:
            centerx = np.subtract(right_fit, 280)
            lane_state.data = 3
        elif left_fit is not None:
            centerx = np.add(left_fit, 280)
            lane_state.data = 1
        else:
            lane_state.data = 0
            
        self.pub_lane_state.publish(lane_state)
        
        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)
        
        if lane_state.data > 0:
            msg_desired_center = Float64()
            msg_desired_center.data = float(centerx.item(350))
            self.pub_lane.publish(msg_desired_center)
            
        if self.pub_image_type == 'compressed':
            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))
        else:
            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(final, 'bgr8'))

    def sliding_window(self, img_w, left_or_right):
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)

        out_img = np.dstack((img_w, img_w, img_w)) * 255

        midpoint = np.int_(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        elif left_or_right == 'right':
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20

        window_height = np.int_(img_w.shape[0] / nwindows)

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

            cv2.rectangle(
                out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            good_lane_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &
                (nonzerox < win_x_high)
                ).nonzero()[0]

            lane_inds.append(good_lane_inds)

            if len(good_lane_inds) > minpix:
                x_current = np.int_(np.mean(nonzerox[good_lane_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            # self.lane_fit_bef = lane_fit

            # save line information
            if left_or_right == 'left':
                self.left_fit = lane_fit
                self.left_fitx = lane_fitx
            else:
                self.right_fit = lane_fit
                self.right_fitx = lane_fitx

        except Exception:
            # lane_fit = self.lane_fit_bef
            lane_fit = self.left_fit if left_or_right == 'left' else self.right_fit
            lane_fitx = self.left_fitx if left_or_right == 'left' else self.right_fitx

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

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

        # 이미지 전처리 및 차선 검출
        processed = self.preprocess_image(cv_image)
        masked = self.create_roi_mask(processed)
        edges = self.detect_edges(masked)
        lines = self.detect_lines(edges)
        
        left_fitx = None
        right_fitx = None
        
        if lines is not None:
            # 좌/우 차선 분리 및 피팅
            left_lines, right_lines = self.separate_lines(lines)
            
            try:
                if len(left_lines) > 0:
                    left_fitx, left_fit = self.fit_from_lines(left_lines, cv_image.shape)
                    if left_fitx is not None:
                        self.mov_avg_left = np.append(self.mov_avg_left, np.array([left_fit]), axis=0)
                        
                if len(right_lines) > 0:
                    right_fitx, right_fit = self.fit_from_lines(right_lines, cv_image.shape)
                    if right_fitx is not None:
                        self.mov_avg_right = np.append(self.mov_avg_right, np.array([right_fit]), axis=0)
                        
            except Exception:
                # 에지 기반 검출 실패 시 sliding window 방식 시도
                if len(left_lines) > 0:
                    left_fitx, left_fit = self.sliding_window(edges, 'left')
                    if left_fit is not None:
                        self.mov_avg_left = np.array([left_fit])
                        
                if len(right_lines) > 0:
                    right_fitx, right_fit = self.sliding_window(edges, 'right')
                    if right_fit is not None:
                        self.mov_avg_right = np.array([right_fit])
        
        # 이동 평균 적용
        MOV_AVG_LENGTH = 5
        
        if self.mov_avg_left.shape[0] > 0:
            left_fit = np.array([
                np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])
            ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            
        if self.mov_avg_right.shape[0] > 0:
            right_fit = np.array([
                np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])
            ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        # 버퍼 크기 제한
        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]
        if self.mov_avg_right.shape[0] > 1000:
            self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]
        
        # 차선 시각화
        self.make_lane(cv_image, left_fitx, right_fitx)



def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
