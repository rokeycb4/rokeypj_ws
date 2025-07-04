#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

class ControlLane(Node):
    def __init__(self):
        super().__init__('control_lane')

        self.sub_lane = self.create_subscription(
            Float64,
            '/lane_center',
            self.cb_follow_lane,
            10
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.cb_get_max_vel,
            10
        )
        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            10
        )

        self.last_error = 0.0
        self.max_vel = 0.1

        self.get_logger().info("ControlLane node has been started.")

    def cb_get_max_vel(self, msg):
        self.max_vel = msg.data

    def cb_follow_lane(self, msg):
        self.get_logger().info("cb_follow_lane called")
        center = msg.data   
        error = center - 500.0

        # PD gains
        kp = 0.0025
        kd = 0.007

        angular_z = kp * error + kd * (error - self.last_error)
        self.last_error = error

        twist = Twist()
        twist.linear.x = min(self.max_vel * ((1 - abs(error) / 500) ** 2.2), 0.05)
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)

        self.pub_cmd_vel.publish(twist)

    def shutdown_callback(self):
        self.get_logger().info("Shutting down. Publishing zero cmd_vel.")
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
