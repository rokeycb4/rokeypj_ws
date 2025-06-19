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

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.subscriber_created = False
        self.active = False

        self.center_value = 500.0  # 기본값 설정

        self.get_logger().info('Node started, waiting 5 seconds before subscribing...')
        self.create_timer(3.0, self.start_subscriber)

    def start_subscriber(self):
        if not self.subscriber_created:
            self.sub_lane = self.create_subscription(
                Float64,
                '/lane_center',
                self.callback_follow_lane,
                1
            )
            self.get_logger().info('Subscribed to /lane_center')
            self.subscriber_created = True

    def callback_follow_lane(self, msg):
        self.center_value = msg.data  # 계속 갱신됨
        # self.get_logger().info(f'{msg.data}')

        if not self.active:
            self.get_logger().info('Received first /detect/lane, starting 2 second movement...')
            self.active = True
            threading.Thread(target=self.drive_for_seconds).start()

    def drive_for_seconds(self):
        start_time = time.time()
        duration = 29  # n초 동안 주행

        while time.time() - start_time < duration:
            error = self.center_value - 500

            twist = Twist()
            twist.linear.x = 0.05
            twist.angular.z = 0.005 * error * 0.1
            self.pub_cmd_vel.publish(twist)

            time.sleep(0.1)  # 10Hz

        self.get_logger().info(f'{duration} seconds done, stopping the robot.')
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
