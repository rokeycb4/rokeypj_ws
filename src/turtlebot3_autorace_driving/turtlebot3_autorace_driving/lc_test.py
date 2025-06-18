#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import time
import threading

class LCTest(Node):
    def __init__(self):
        super().__init__('lc_test')

        self.pub_cmd_vel = self.create_publisher(Twist, '/control/cmd_vel', 1)
        self.subscriber_created = False
        self.active = False

        self.get_logger().info('Node started, waiting 3 seconds before subscribing...')
        self.create_timer(8.0, self.start_subscriber)

    def start_subscriber(self):
        if not self.subscriber_created:
            self.sub_lane = self.create_subscription(
                Float64,
                '/detect/lane',
                self.callback_follow_lane,
                1
            )
            self.get_logger().info('Subscribed to /detect/lane')
            self.subscriber_created = True

    def callback_follow_lane(self, msg):
        if not self.active:
            self.get_logger().info('Received first lane center, starting 5 second movement...')
            self.active = True
            threading.Thread(target=self.drive_for_seconds, args=(msg.data,)).start()

    def drive_for_seconds(self, center_value):
        start_time = time.time()
        duration = 2  # 5초 동안 주행

        while time.time() - start_time < duration:
            error = center_value - 500

            twist = Twist()
            twist.linear.x = 0.1
            twist.angular.z = -0.005 * error
            self.pub_cmd_vel.publish(twist)

            time.sleep(0.1)  # 10Hz

        self.get_logger().info('5 seconds done, stopping the robot.')
        self.stop_robot()

    def stop_robot(self):
        twist = Twist()  # 속도 0
        self.pub_cmd_vel.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LCTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
