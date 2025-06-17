#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import IntegerRange, ParameterDescriptor, SetParametersResult
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

class ImageProjection(Node):
    def __init__(self):
        super().__init__('image_projection')

        # 파라미터 범위 설정 (이미지 해상도 기준으로 수정)
        descriptor_top = ParameterDescriptor(
            description='projection range top',
            integer_range=[IntegerRange(from_value=0, to_value=320, step=1)]
        )
        descriptor_bottom = ParameterDescriptor(
            description='projection range bottom',
            integer_range=[IntegerRange(from_value=0, to_value=640, step=1)]
        )

        # 파라미터 선언 (초기값도 해상도에 맞춰 조정)
        self.declare_parameters(
            '',
            [
                ('camera.extrinsic_camera_calibration.top_x', 86, descriptor_bottom),
                ('camera.extrinsic_camera_calibration.top_y', 0, descriptor_top),
                ('camera.extrinsic_camera_calibration.bottom_x', 169, descriptor_bottom),
                ('camera.extrinsic_camera_calibration.bottom_y', 120, descriptor_top),
                ('is_extrinsic_camera_calibration_mode', False)
            ]
        )

        self.load_parameters()

        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.param_callback)

        # 이미지 타입: in캘리브레이션에 맞게 raw로 구독
        self.cvBridge = CvBridge()
        self.sub_image = self.create_subscription(
            CompressedImage, '/camera/image_input', self.cbImageProjection, 1
        )

        self.pub_image_projected = self.create_publisher(Image, '/camera/image_output', 1)

        if self.is_calibration_mode:
            self.pub_image_calib = self.create_publisher(Image, '/camera/image_calib', 1)

    def load_parameters(self):
        p = self.get_parameter
        self.top_x = p('camera.extrinsic_camera_calibration.top_x').get_parameter_value().integer_value
        self.top_y = p('camera.extrinsic_camera_calibration.top_y').get_parameter_value().integer_value
        self.bottom_x = p('camera.extrinsic_camera_calibration.bottom_x').get_parameter_value().integer_value
        self.bottom_y = p('camera.extrinsic_camera_calibration.bottom_y').get_parameter_value().integer_value
        self.is_calibration_mode = p('is_extrinsic_camera_calibration_mode').get_parameter_value().bool_value

    def param_callback(self, parameters):
        for param in parameters:
            if param.name.endswith('top_x'):
                self.top_x = param.value
            elif param.name.endswith('top_y'):
                self.top_y = param.value
            elif param.name.endswith('bottom_x'):
                self.bottom_x = param.value
            elif param.name.endswith('bottom_y'):
                self.bottom_y = param.value

        self.get_logger().info(
            f'Updated projection params: top({self.top_x}, {self.top_y}), bottom({self.bottom_x}, {self.bottom_y})')
        return SetParametersResult(successful=True)

    def cbImageProjection(self, msg_img):
        cv_image_original = self.cvBridge.imgmsg_to_cv2(msg_img, 'bgr8')
        cv_image_original = cv2.GaussianBlur(cv_image_original, (5, 5), 0)

        # 소스 좌표 (이미지 중앙 기준 좌표 계산)
        pts_src = np.array([
            [320 - self.top_x, 240 - self.top_y],
            [320 + self.top_x, 240 - self.top_y],
            [320 + self.bottom_x, 240 + self.bottom_y],
            [320 - self.bottom_x, 240 + self.bottom_y]
        ])

        # 타겟 좌표 (결과 이미지 크기는 기존 1000x600 유지)
        pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])
        h, _ = cv2.findHomography(pts_src, pts_dst)
        cv_image_homography = cv2.warpPerspective(cv_image_original, h, (1000, 600))

        # 삼각형 블랙 마스킹 (기존 유지)
        triangle1 = np.array([[0, 599], [0, 340], [200, 599]], np.int32)
        triangle2 = np.array([[999, 599], [999, 340], [799, 599]], np.int32)
        cv_image_homography = cv2.fillPoly(cv_image_homography, [triangle1, triangle2], (0, 0, 0))

        # calibration 모드면 박스 표시
        if self.is_calibration_mode:
            calib_img = np.copy(cv_image_original)
            cv2.line(calib_img, (320 - self.top_x, 240 - self.top_y), (320 + self.top_x, 240 - self.top_y), (0, 0, 255), 1)
            cv2.line(calib_img, (320 - self.bottom_x, 240 + self.bottom_y), (320 + self.bottom_x, 240 + self.bottom_y), (0, 0, 255), 1)
            cv2.line(calib_img, (320 + self.bottom_x, 240 + self.bottom_y), (320 + self.top_x, 240 - self.top_y), (0, 0, 255), 1)
            cv2.line(calib_img, (320 - self.bottom_x, 240 + self.bottom_y), (320 - self.top_x, 240 - self.top_y), (0, 0, 255), 1)
            self.pub_image_calib.publish(self.cvBridge.cv2_to_imgmsg(calib_img, 'bgr8'))

        # 결과 발행
        self.pub_image_projected.publish(self.cvBridge.cv2_to_imgmsg(cv_image_homography, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = ImageProjection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
