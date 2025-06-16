from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aruco_yolo',
            executable='compressed_image_pub',
            name='compressed_image_pub',
            output='screen'
        ),
        
        Node(
            package='aruco_yolo',
            executable='aruco_detector',
            name='aruco_detector',
            output='screen'
        ),
        
        Node(
            package='aruco_yolo',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen'
        )
    ])