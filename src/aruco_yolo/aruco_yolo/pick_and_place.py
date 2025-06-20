import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from aruco_msgs.msg import MarkerArray
from turtlebot_cosmo_interface.srv import MoveitControl
from geometry_msgs.msg import Pose

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

class AutoPickPlaceNode(Node):
    def __init__(self):
        super().__init__('auto_pick_place_node')

        self.moveit_client = self.create_client(MoveitControl, 'moveit_control')
        while not self.moveit_client.wait_for_service(timeout_sec=1.0):
            self.log_info('MoveIt 서비스 대기 중...')

        self.subscription = self.create_subscription(
            MarkerArray,
            'detected_markers',
            self.marker_callback,
            10)

        self.processing = False
        self.marker_pose = None
        self.timer_handle = None

    def log_info(self, msg):
        self.get_logger().info(f"{GREEN}{msg}{RESET}")

    def log_warn(self, msg):
        self.get_logger().warn(f"{YELLOW}{msg}{RESET}")

    def log_error(self, msg):
        self.get_logger().error(f"{RED}{msg}{RESET}")

    def marker_callback(self, msg):
        if self.processing or len(msg.markers) == 0:
            return

        marker = msg.markers[0]
        self.marker_pose = marker.pose.pose

        self.log_info(f"마커 위치 수신: x={self.marker_pose.position.x:.3f}, y={self.marker_pose.position.y:.3f}, z={self.marker_pose.position.z:.3f}")

        self.processing = True
        self.execute_pick_and_place()

    def send_moveit_request(self, cmd, target, callback=None):
        req = MoveitControl.Request()
        req.cmd = cmd
        if cmd == 0:
            req.waypoints = target
        else:
            req.posename = target

        self.log_info(f"MoveitControl 서비스 요청: cmd={cmd}, target={target}")
        future = self.moveit_client.call_async(req)

        def _handle_response(fut):
            if fut.result() is not None and fut.result().response:
                self.log_info(f"서비스 성공: cmd={cmd}, target={target}")
                if callback:
                    callback(True)
            else:
                self.log_error(f"서비스 실패: cmd={cmd}, target={target}")
                if callback:
                    callback(False)

        future.add_done_callback(_handle_response)

    def schedule_step(self, delay_sec, func):
        if self.timer_handle:
            self.timer_handle.cancel()
        self.timer_handle = self.create_timer(delay_sec, lambda: (self.timer_handle.cancel(), func())[1])

    def execute_pick_and_place(self):
        self.log_info("Pick & Place 작업 시작")
        self.send_moveit_request(2, "open", self.step_2_box_front)

    def step_2_box_front(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "02_box_front", self.step_3_move_to_box))

    def step_3_move_to_box(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "03_move_to_box", self.step_4_close_gripper))

    def step_4_close_gripper(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(2, "close", self.step_5_lift_up))

    def step_5_lift_up(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "04_move_up", self.step_6_conveyor_up))

    def step_6_conveyor_up(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "05_conveyor_up", self.step_7_conveyor_down))

    def step_7_conveyor_down(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "06_conveyor_down", self.step_8_open_gripper))

    def step_8_open_gripper(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(2, "open", self.step_9_conveyor_up))

    def step_9_conveyor_up(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "07_conveyor_up", self.step_10_return_home))

    def step_10_return_home(self, success):
        if not success:
            self.processing = False
            return
        self.schedule_step(2.0, lambda: self.send_moveit_request(1, "lane_tracking_01", self.finish_sequence))

    def finish_sequence(self, success):
        if success:
            self.log_info("Pick & Place 작업 완료")
        else:
            self.log_error("복귀 실패")
        self.processing = False

def main(args=None):
    rclpy.init(args=args)
    node = AutoPickPlaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()