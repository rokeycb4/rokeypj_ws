#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

import numpy as np
from scipy.signal import resample
import openwakeword
from openwakeword.model import Model

from ament_index_python.packages import get_package_share_directory
from . import MicController
import os


class WakeupWord:
    def __init__(self, buffer_size, node):
        self.node = node
        self.buffer_size = buffer_size
        self.stream = None

        # 모델 경로 설정
        package_path = get_package_share_directory("turtlebot3_autorace_detect")
        MODEL_NAME = "멈처.tflite"
        MODEL_PATH = os.path.join(package_path, f"resource/{MODEL_NAME}")

        openwakeword.utils.download_models()

        # 모델 초기화
        try:
            self.model = Model(wakeword_models=[MODEL_PATH])
            self.model_name = "멈처"  # 학습 시 지정한 웨이크워드 키
            self.node.get_logger().info(f"Loaded model from: {MODEL_PATH}")
        except Exception as e:
            self.node.get_logger().error(f"Failed to load model: {e}")
            raise

    def is_wakeup(self):
        try:
            raw_audio = self.stream.read(self.buffer_size, exception_on_overflow=False)
        except Exception as e:
            self.node.get_logger().warn(f"Audio read error: {e}")
            return False

        if not raw_audio or len(raw_audio) == 0:
            self.node.get_logger().warn("Audio chunk is empty.")
            return False

        audio_chunk = np.frombuffer(raw_audio, dtype=np.int16)

        if audio_chunk.size == 0:
            self.node.get_logger().warn("Audio chunk is empty after np.frombuffer.")
            return False

        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))

        if audio_chunk.size == 0:
            self.node.get_logger().warn("Resampled audio chunk is empty.")
            return False

        self.node.get_logger().info(f"Audio chunk max: {np.max(audio_chunk):.4f}, min: {np.min(audio_chunk):.4f}")

        try:
            outputs = self.model.predict(audio_chunk, threshold=0.1)
            self.node.get_logger().info(f"Raw model outputs: {outputs}")
            confidence = outputs.get(self.model_name, 0.0)
            self.node.get_logger().info(f"confidence: {confidence:.3f}")
        except Exception as e:
            self.node.get_logger().error(f"Prediction error: {e}")
            return False

        if confidence > 0.25:
            self.node.get_logger().info("Wakeword detected!")
            return True
        return False

    def set_stream(self, stream):
        self.stream = stream


class StopWordNode(Node):
    def __init__(self):
        super().__init__('detect_stop_voice')
        self.publisher_ = self.create_publisher(Bool, '/detect/stopword', 10)

        self.mic = MicController.MicController()
        self.mic.open_stream()

        self.wakeup = WakeupWord(self.mic.config.buffer_size, self)
        self.wakeup.set_stream(self.mic.stream)

        self.get_logger().info("StopWordNode initialized and listening...")

        self.timer = self.create_timer(0.3, self.check_wakeword)

    def check_wakeword(self):
        try:
            if self.wakeup.is_wakeup():
                msg = Bool()
                msg.data = True
                self.publisher_.publish(msg)
                self.get_logger().info(">>> Published stopword: True")
        except Exception as e:
            self.get_logger().error(f"Error during wakeword check: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StopWordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt: shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
