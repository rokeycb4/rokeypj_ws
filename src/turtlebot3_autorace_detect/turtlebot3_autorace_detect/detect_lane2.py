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

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(from_value=0, to_value=179, step=1)]
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            description='saturation and lightness range',
            integer_range=[IntegerRange(from_value=0, to_value=255, step=1)]
        )

        self.declare_parameters('', [
            ('detect.lane.white.hue_l', 0, parameter_descriptor_hue),
            ('detect.lane.white.hue_h', 179, parameter_descriptor_hue),
            ('detect.lane.white.saturation_l', 0, parameter_descriptor_saturation_lightness),
            ('detect.lane.white.saturation_h', 70, parameter_descriptor_saturation_lightness),
            ('detect.lane.white.lightness_l', 105, parameter_descriptor_saturation_lightness),
            ('detect.lane.white.lightness_h', 255, parameter_descriptor_saturation_lightness),
            ('detect.lane.yellow.hue_l', 10, parameter_descriptor_hue),
            ('detect.lane.yellow.hue_h', 127, parameter_descriptor_hue),
            ('detect.lane.yellow.saturation_l', 70, parameter_descriptor_saturation_lightness),
            ('detect.lane.yellow.saturation_h', 255, parameter_descriptor_saturation_lightness),
            ('detect.lane.yellow.lightness_l', 95, parameter_descriptor_saturation_lightness),
            ('detect.lane.yellow.lightness_h', 255, parameter_descriptor_saturation_lightness),
            ('is_detection_calibration_mode', False)
        ])

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
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_type = 'compressed'
        self.pub_image_type = 'compressed'

        # # 여기: launch파일 remapping을 통해 외부 토픽 이름이 연결됨
        # if self.sub_image_type == 'compressed':
        #     self.sub_image_original = self.create_subscription(
        #         CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1)
        # else:
        #     self.sub_image_original = self.create_subscription(
        #         Image, '/detect/image_input', self.cbFindLane, 1)

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                #CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                CompressedImage, '/image_raw/compressed', self.cbFindLane, 1                
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )


        self.pub_image_lane = self.create_publisher(
            CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)
        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()
        self.counter = 1

    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            if param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
        return SetParametersResult(successful=True)

    def cbFindLane(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # 간단히: 그려서 퍼블리시 (실제로는 기존 로직 넣으면 됨)
        final = cv_image.copy()

        self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
