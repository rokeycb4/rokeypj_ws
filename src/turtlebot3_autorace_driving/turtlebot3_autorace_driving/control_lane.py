from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool, UInt8
import rclpy
from rclpy.node import Node
import time

class ControlLane(Node):

    # 상태
    STATE_RUN = 0
    STATE_STOP = 1

    def __init__(self):
        super().__init__('control_lane')

        # 구독자 설정
        self.sub_lane = self.create_subscription(
            Float64,
            '/lane_center',
            self.callback_follow_lane,
            1
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/max_vel',
            self.callback_get_max_vel,
            1
        )
        self.sub_stopline = self.create_subscription(
            Bool,
            '/detect/stopline',
            self.callback_stopline,
            1
        )
        self.sub_lane_state = self.create_subscription(
            UInt8,
            '/lane_state',
            self.callback_lane_state,
            1
        )

        # 퍼블리셔 설정
        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/cmd_vel',
            1
        )

        # PD 제어 관련 변수
        self.last_error = 0
        self.MAX_VEL = 0.1

        # 상태 변수 초기화
        self.state = self.STATE_RUN
        self.prev_lane_state = 2  # 초기에는 정상 주행 상태라고 가정
        self.curr_lane_state = 2
        self.last_valid_center = 640.0  # 마지막 유효한 center 값 저장

    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    def callback_follow_lane(self, desired_center):
        # lane_state에 의해 주행이 방해되면 아무 동작도 하지 않음
        if self.state != self.STATE_RUN:
            return

        center = desired_center.data
        self.last_valid_center = center  # 마지막 유효한 center 값 저장
        error = (center - 640) * 0.5

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

<<<<<<< HEAD
=======

>>>>>>> 5b60e44b30c4058db80e34018832c29e0158f50c
        twist = Twist()
        # 선형 속도: error에 따라 속도를 조정 (최대 0.05 제한)
        twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 640, 0) ** 2.2), 0.03)
        twist.angular.z = -max(angular_z, -0.3) if angular_z < 0 else -min(angular_z, 0.3)
        self.pub_cmd_vel.publish(twist)

    def callback_stopline(self, msg):
        if msg.data and self.state != self.STATE_STOP:
            self.get_logger().info('Stopline detected. Stopping the robot.')
            self.state = self.STATE_STOP
            twist = Twist()  # 모든 속도를 0으로 설정
            self.pub_cmd_vel.publish(twist)

    def callback_lane_state(self, msg):
        # 상태 전이
        self.prev_lane_state = self.curr_lane_state
        self.curr_lane_state = msg.data

        if self.prev_lane_state in [1, 3] and self.curr_lane_state == 0:
            self.get_logger().warn(f"Lost lane after only {self.prev_lane_state} side. Using last valid center.")
            self.hold_center_temporarily()
        elif self.prev_lane_state == 2 and self.curr_lane_state == 0:
            self.get_logger().warn("Both lanes lost. Stopping.")
            self.stop_vehicle()
        elif self.prev_lane_state == 0 and self.curr_lane_state == 2:
            self.get_logger().info("Recovered lane after losing both lanes. Resuming drive.")
            self.state = self.STATE_RUN

    def hold_center_temporarily(self):
        # 이전 center 값으로 약한 유지 조향
        error = self.last_valid_center - 640
        Kp = 0.002
        twist = Twist()
        twist.linear.x = 0.03
        twist.angular.z = -Kp * error
        self.pub_cmd_vel.publish(twist)

    def stop_vehicle(self):
        twist = Twist()
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
