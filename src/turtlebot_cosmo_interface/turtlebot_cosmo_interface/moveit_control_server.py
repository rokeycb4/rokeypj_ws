#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from turtlebot_cosmo_interface.srv import MoveitControl
from geometry_msgs.msg import PoseArray

class MoveitControlServer(Node):
    def __init__(self):
        super().__init__('moveit_control_server')

        # 서비스 등록
        self.srv = self.create_service(MoveitControl, 'moveit_control', self.handle_request)
        self.get_logger().info('✅ MoveitControl 서비스 서버 시작됨')

        # 그리퍼 제어 퍼블리셔 (예시: Float64 값으로 열고 닫기)
        self.gripper_pub = self.create_publisher(Float64, '/gripper_controller/command', 10)

    def handle_request(self, request, response):
        cmd = request.cmd
        target = request.posename
        self.get_logger().info(f"📨 서비스 요청 수신: cmd={cmd}, target={target}")

        if cmd == 1:
            # predefined pose name으로 move_group에게 이동 명령
            self.get_logger().info(f"MoveIt으로 pose '{target}' 이동 시도")

            # 예시: target_pose_dict에서 joint 값 조회
            if target in self.pose_dict:
                joint_goal = self.pose_dict[target]
                move_group.go(joint_goal, wait=True)
                move_group.stop()
                self.get_logger().info(f"✅ pose '{target}' 이동 완료")
                response.response = True
            else:
                self.get_logger().error(f"❌ 정의되지 않은 pose name: {target}")
                response.response = False

        elif cmd == 2:
            if target == 'open':
                self.gripper_pub.publish(Float64(data=0.0))  # 열기
                self.get_logger().info("🔓 그리퍼 열기 완료")
                response.response = True
            elif target == 'close':
                self.gripper_pub.publish(Float64(data=1.0))  # 닫기
                self.get_logger().info("🔒 그리퍼 닫기 완료")
                response.response = True
            else:
                self.get_logger().error(f"❗ 알 수 없는 그리퍼 명령: {target}")
                response.response = False

        
        else:
            self.get_logger().warn(f"⚠️ cmd={cmd} 은 현재 처리되지 않음 (posename={target})")
            response.response = False

        return response

def main(args=None):
    rclpy.init(args=args)
    node = MoveitControlServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
