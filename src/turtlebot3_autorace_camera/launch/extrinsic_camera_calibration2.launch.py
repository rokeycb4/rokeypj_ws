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
        description='calibration mode type [True, False]')
    calibration_mode = LaunchConfiguration('calibration_mode')

    compensation_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_camera'),
        'calibration',
        'extrinsic_calibration',
        'compensation.yaml'
    )

    projection_param = os.path.join(
        get_package_share_directory('turtlebot3_autorace_camera'),
        'calibration',
        'extrinsic_calibration',
        'projection.yaml'
    )

    image_projection_node = Node(
        package='turtlebot3_autorace_camera',
        executable='image_projection',
        namespace='camera',
        name='image_projection',
        output='screen',
        parameters=[
            projection_param,
            {'is_extrinsic_camera_calibration_mode': calibration_mode}
        ],
        remappings=[
            ('/camera/image_input', '/camera/image'),
            
            ('/camera/image_output', '/camera/image_projected'),
            ('/camera/image_output/compressed', '/camera/image_projected/compressed'),
            ('/camera/image_calib', '/camera/image_extrinsic_calib'),
            ('/camera/image_calib/compressed', '/camera/image_extrinsic_calib/compressed')
        ],
    )

    image_compensation_node = Node(
        package='turtlebot3_autorace_camera',
        executable='image_compensation',
        namespace='camera',
        name='image_compensation',
        output='screen',
        parameters=[{
            'is_extrinsic_camera_calibration_mode': calibration_mode
        }, compensation_param],

    remappings=[
        ('/camera/image_input', '/camera/image'),
        ('/camera/image_input/compressed', '/camera/image'),
        
        ('/camera/image_output', '/camera/image_compensated'),
        ('/camera/image_output/compressed', '/camera/image_compensated/compressed')
    ]

    )

    return LaunchDescription([
        calibration_mode_arg,
        image_projection_node,
        image_compensation_node
    ])
