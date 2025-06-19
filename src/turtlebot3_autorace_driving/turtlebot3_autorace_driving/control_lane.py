from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool


class ControlLane(Node):

    # state
    STATE_RUN = 0
    STATE_STOP = 1

    def __init__(self):
        super().__init__('control_lane')

        self.sub_lane = self.create_subscription(
            Float64,
            '/lane_center',
            self.callback_follow_lane,
            1
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.callback_get_max_vel,
            1
        )
        self.sub_stopline = self.create_subscription(
            Bool,
            '/detect/stopline',
            self.callback_stopline,
            1
        )

        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            1
        )

        # PD control related variables
        self.last_error = 0
        self.MAX_VEL = 0.1

        # Initial state
        self.state = self.STATE_RUN

    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    def callback_follow_lane(self, desired_center):

        self.get_logger().info(f'/lane_center: {desired_center.data}')


        if self.state != self.STATE_RUN:
            return

        center = desired_center.data
        error = center - 500

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        twist = Twist()
        # Linear velocity: adjust speed based on error (maximum 0.05 limit)
        twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05)
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.pub_cmd_vel.publish(twist)

    def callback_stopline(self, msg):
        if msg.data and self.state != self.STATE_STOP:
            self.get_logger().info('Stopline detected. Stopping the robot.')
            self.state = self.STATE_STOP
            twist = Twist()  # All velocities zero
            self.pub_cmd_vel.publish(twist)

    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
