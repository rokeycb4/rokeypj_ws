# 기존코드에 토픽만 바꾼거

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # calibration_mode 파라미터 선언
    calibration_mode_arg = DeclareLaunchArgument(
        'calibration_mode',
        default_value='False',
        description='Mode type [calibration, action]'
    )
    calibration_mode = LaunchConfiguration('calibration_mode')

    # raw_mode 파라미터 선언 (추가된 부분)
    raw_mode_arg = DeclareLaunchArgument(
        'raw_mode',
        default_value='False',
        description='Use raw image topic'
    )
    raw_mode = LaunchConfiguration('raw_mode')

    # 파라미터 파일 경로
    detect_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_detect'),
        'param',
        'lane',
        'lane.yaml'
    )

    # 노드 실행부
    detect_lane_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_lane3',
        name='detect_lane3',
        output='screen',
        parameters=[
            {'is_detection_calibration_mode': calibration_mode},
            {'raw_mode': raw_mode},
            detect_param
        ]
    )

    return LaunchDescription([
        calibration_mode_arg,
        raw_mode_arg,
        detect_lane_node,
    ])
