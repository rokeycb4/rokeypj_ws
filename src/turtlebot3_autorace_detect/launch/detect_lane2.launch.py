#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    calibration_mode_arg = DeclareLaunchArgument(
        'calibration_mode',
        default_value='False',
        description='Mode type [calibration, action]'
    )
    calibration_mode = LaunchConfiguration('calibration_mode')

    detect_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_detect'),
        'param',
        'lane',
        'lane.yaml'
    )

    detect_lane_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_lane2',
        name='detect_lane2',
        output='screen',
        parameters=[
            {'is_detection_calibration_mode': calibration_mode},
            detect_param
        ],
        # remappings 제거
    )

    return LaunchDescription([
        calibration_mode_arg,
        detect_lane_node,
    ])
