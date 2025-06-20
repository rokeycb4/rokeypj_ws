import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from aruco_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
import math

class TurtlebotArucoDetector(Node):
    def __init__(self):
        super().__init__('turtlebot_aruco_detector')
        
        # QoS 프로필 설정 (압축 이미지 및 일반 이미지 모두 대응)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 파라미터 선언
        self.declare_parameter('marker_dict', 'DICT_6x6_1000')
        self.declare_parameter('marker_size', 0.05)
        self.declare_parameter('min_distance', 0.1)
        self.declare_parameter('max_distance', 2.0)
        self.declare_parameter('allowed_marker_ids', [0, 1, 2, 3, 4, 5, 6, 7])
        
        # 파라미터 값 읽기
        marker_dict_name = self.get_parameter('marker_dict').get_parameter_value().string_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        self.min_distance = self.get_parameter('min_distance').get_parameter_value().double_value
        self.max_distance = self.get_parameter('max_distance').get_parameter_value().double_value
        self.allowed_marker_ids = self.get_parameter('allowed_marker_ids').get_parameter_value().integer_array_value
        
        # 마커 사전 동적 선택
        marker_dicts = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000
        }
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            marker_dicts.get(marker_dict_name, cv2.aruco.DICT_4X4_50)
        )
        
        # 카메라 토픽 구독 (압축 및 일반 이미지 모두 지원)
        self.image_subscription = self.create_subscription(
            Image, 
            '/camera/image_raw',  # 기본 이미지 토픽
            self.camera_callback, 
            qos_profile
        )
        
        self.compressed_image_subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',  # 압축 이미지 토픽
            self.compressed_camera_callback, 
            qos_profile
        )
        
        # ArUco 마커 정보 발행을 위한 퍼블리셔 추가
        self.marker_publisher = self.create_publisher(
            MarkerArray, 
            'detected_markers', 
            10
        )
        
        # 마커 표시 이미지 발행 퍼블리셔
        self.marker_image_publisher = self.create_publisher(
            Image,
            'aruco_markers/image_raw',
            10
        )
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # 캘리브레이션 파라미터 로드
        self.camera_matrix, self.dist_coeffs = self.load_camera_parameters()
    
    def load_camera_parameters(self):
        # 캘리브레이션 파일 경로(환경에 맞게 설정)
        calibration_file = '/home/rokey-jw/rokeypj_ws/src/aruco_yolo/config/calibration_params.yaml'

        try:
            with open(calibration_file, 'r') as f:
                data = yaml.safe_load(f)
                camera_matrix = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
                dist_coeffs = np.array(data["distortion_coefficients"]["data"], dtype=np.float32)
                
                # 로그로 캘리브레이션 정보 출력
                self.get_logger().info(f"Camera Matrix:\n{camera_matrix}")
                self.get_logger().info(f"Distortion Coefficients: {dist_coeffs}")
        except Exception as e:
            self.get_logger().warning(f"캘리브레이션 파일 로드 실패: {e}")
            # 기본값 사용
            camera_matrix = np.array([[1050.64415, 0, 636.22318], 
                                      [0, 1050.26622, 360.07389], 
                                      [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([0.051971, 0.053343, 0.024273, 0.000262, 0.000000], dtype=np.float32)
        
        return camera_matrix, dist_coeffs
    
    def detect_markers(self, image):
        # 디텍터 파라미터 상세 설정
        parameters = cv2.aruco.DetectorParameters()
        
        # 파라미터 세부 조정
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.03
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.polygonalApproxAccuracyRate = 0.03
        parameters.minCornerDistanceRate = 0.05
        parameters.minMarkerDistanceRate = 0.05
        parameters.minDistanceToBorder = 3
        parameters.maxErroneousBitsInBorderRate = 0.1
        parameters.errorCorrectionRate = 0.4
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # ArUco 디텍터 생성
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 마커 감지
        corners, ids, _ = detector.detectMarkers(gray)
        
        # MarkerArray 메시지 생성
        marker_array = MarkerArray()
        
        if ids is not None and len(ids) > 0:

            filtered_indices = [i for i, id_ in enumerate(ids) if int(id_[0]) in self.allowed_marker_ids]
            if not filtered_indices:
                self.get_logger().info("허용된 ID의 마커가 없습니다.")
                return image

            # ID와 corners 필터링
            ids = [ids[i] for i in filtered_indices]
            corners = [corners[i] for i in filtered_indices]
            
            # NumPy 배열로 다시 변환
            ids = np.array(ids)

            # 마커 그리기 (초록색 테두리)
            cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
            
            self.get_logger().info(f"총 {len(ids)}개의 마커 감지됨:")
            
            for i, marker_id in enumerate(ids):
                # 마커의 4개 코너 좌표를 객체 포인트로 사용
                object_points = np.array([
                    [0, 0, 0],
                    [self.marker_size, 0, 0],
                    [self.marker_size, self.marker_size, 0],
                    [0, self.marker_size, 0]
                ], dtype=np.float32)
                
                # 마커 위치 추정
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    corners[i],
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                if success:
                    # 거리 계산 (정확한 3D 거리)
                    distance = np.linalg.norm(tvec)
                    # distance는 카메라 원점에서 마커 중심까지의 3D 유클리드 거리입니다.
                    
                    # 마커 필터링 (ID 및 거리 조건)
                    if (int(marker_id[0]) in self.allowed_marker_ids and 
                        self.min_distance < distance < self.max_distance):
                        
                        # 회전각도 계산
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        
                        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  
                                     rotation_matrix[1,0] * rotation_matrix[1,0])
                        
                        singular = sy < 1e-6
                        
                        if not singular:
                            yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / np.pi
                            pitch = np.arctan2(-rotation_matrix[2,0], sy) * 180 / np.pi
                            roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]) * 180 / np.pi
                        else:
                            yaw = np.arctan2(-rotation_matrix[0,1], rotation_matrix[1,1]) * 180 / np.pi
                            pitch = np.arctan2(-rotation_matrix[2,0], sy) * 180 / np.pi
                            roll = 0
                        
                        # 마커 메시지 생성
                        marker_msg = Marker()
                        marker_msg.id = int(marker_id[0])
                        
                        # PoseStamped로 위치 정보 설정
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = 'camera_link'
                        
                        # 위치 정보 설정
                        pose_msg.pose.position.x = tvec[0][0]
                        pose_msg.pose.position.y = tvec[1][0]
                        pose_msg.pose.position.z = tvec[2][0]
                        
                        marker_msg.pose = pose_msg
                        # roll, pitch, yaw를 라디안으로 변환하여 rot에 할당
                        marker_msg.rot.x = math.radians(roll)
                        marker_msg.rot.y = math.radians(pitch)
                        marker_msg.rot.z = math.radians(yaw)
                        marker_msg.distance = float(distance) 
                        marker_array.markers.append(marker_msg)
                        
                        # 거리 값 로그로 출력
                        self.get_logger().info(
                            f"[마커 {int(marker_id[0])}] Distance(3D norm): {distance:.3f}m"
                        )
                        
                        # 이미지에 정보 표시
                        info_texts = [
                            f"ID: {int(marker_id[0])}",
                            f"Pos: ({tvec[0][0]*1000:.2f}, {tvec[1][0]*1000:.2f}, {tvec[2][0]*1000:.2f})mm",
                            f"Dist: {distance:.2f}m",  # 3D 거리
                            f"Rot: ({yaw:.2f}, {pitch:.2f}, {roll:.2f})deg"
                        ]
                        
                        colors = [
                            (255, 128, 0),  # 하늘색
                            (0, 255, 0),    # 초록색
                            (0, 0, 255),    # 파란색
                            (255, 0, 0)     # 빨간색
                        ]
                        
                        c = corners[i][0]
                        center_x = int(np.mean(c[:, 0]))
                        center_y = int(np.mean(c[:, 1]))
                        
                        for idx, (text, color) in enumerate(zip(info_texts, colors)):
                            cv2.putText(image, text, 
                                        (center_x, center_y + idx * 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, color, 2)
            
            # MarkerArray 토픽 발행
            self.marker_publisher.publish(marker_array)
        else:
            self.get_logger().info("마커를 찾을 수 없습니다.")
        
        return image
    
    def camera_callback(self, msg):
        # ROS 이미지 메시지를 OpenCV 이미지로 변환
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return
        
        # ArUco 마커 감지
        annotated_image = self.detect_markers(cv_image)
        
        # 마커 표시된 이미지를 ROS 토픽으로 발행
        try:
            marker_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.marker_image_publisher.publish(marker_image_msg)
        except Exception as e:
            self.get_logger().error(f"이미지 발행 실패: {e}")
        
        # 디버깅을 위해 이미지 표시
        # cv2.imshow("Turtlebot ArUco Marker Detection", annotated_image)
        cv2.waitKey(1)
    
    def compressed_camera_callback(self, msg):
        # ROS 압축 이미지 메시지를 OpenCV 이미지로 변환
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"압축 이미지 변환 실패: {e}")
            return
        
        # ArUco 마커 감지
        annotated_image = self.detect_markers(cv_image)
        
        # 마커 표시된 이미지를 ROS 토픽으로 발행
        try:
            marker_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            self.marker_image_publisher.publish(marker_image_msg)
        except Exception as e:
            self.get_logger().error(f"이미지 발행 실패: {e}")
        
        # 디버깅을 위해 이미지 표시
        # cv2.imshow("Turtlebot ArUco Marker Detection", annotated_image)
        # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotArucoDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 