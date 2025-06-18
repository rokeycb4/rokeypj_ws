import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class TurtlebotArucoDetector(Node):
    def __init__(self):
        super().__init__('turtlebot_aruco_detector')
        
        # QoS 프로필 설정 (압축 이미지 및 일반 이미지 모두 대응)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
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
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # 캘리브레이션 파라미터 로드
        self.camera_matrix, self.dist_coeffs = self.load_camera_parameters()
        
        # 마커 크기 (미터)
        self.marker_size = 0.04
    
    def load_camera_parameters(self):
        # 캘리브레이션 파일 경로
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
        # 가장 일반적으로 사용되는 마커 사전 선택
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # 마커 감지
        corners, ids, _ = detector.detectMarkers(image)
        
        if ids is not None and len(ids) > 0:
            # 마커 그리기 (초록색 테두리)
            cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
            
            for i, marker_id in enumerate(ids):
                # 마커 중심점 계산
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                
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
                    # 거리 계산
                    distance = np.linalg.norm(tvec)
                    
                    # 회전각도 계산
                    rvec_deg = np.degrees(rvec).flatten()
                    
                    # 로그 출력
                    self.get_logger().info(f"Marker ID: {marker_id[0]}")
                    self.get_logger().info(f"Position: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})")
                    self.get_logger().info(f"Rotation: ({rvec_deg[0]:.2f}, {rvec_deg[1]:.2f}, {rvec_deg[2]:.2f})deg")
                    self.get_logger().info(f"Distance: {distance:.2f}m")
                    
                    # 이미지에 정보 표시
                    info_texts = [
                        f"ID: {marker_id[0]}",
                        f"Pos: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})",
                        f"Dist: {distance:.2f}m",
                        f"Rot: ({rvec_deg[0]:.2f}, {rvec_deg[1]:.2f}, {rvec_deg[2]:.2f})deg"
                    ]
                    
                    colors = [
                        (255, 128, 0),  # 하늘색
                        (0, 255, 0),    # 초록색
                        (0, 0, 255),    # 파란색
                        (255, 0, 0)     # 빨간색
                    ]
                    
                    for idx, (text, color) in enumerate(zip(info_texts, colors)):
                        cv2.putText(image, text, 
                                    (center_x, center_y + idx * 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, color, 2)
        
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
        
        # 디버깅을 위해 이미지 표시
        cv2.imshow("Turtlebot ArUco Marker Detection", annotated_image)
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
        
        # 디버깅을 위해 이미지 표시
        cv2.imshow("Turtlebot ArUco Marker Detection", annotated_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotArucoDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C로 노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 