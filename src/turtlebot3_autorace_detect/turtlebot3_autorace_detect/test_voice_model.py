#!/usr/bin/env python3
import os
import sys
import select
import numpy as np
from scipy.signal import resample
import openwakeword
from openwakeword.model import Model
import pyaudio
import tty
import termios

# === 수정할 부분 ===
MODEL_NAME = "멈처.tflite"  # ← 테스트할 .tflite 파일명
PACKAGE_NAME = "turtlebot3_autorace_detect"
# ===================

# from ament_index_python.packages import get_package_share_directory
# package_path = get_package_share_directory(PACKAGE_NAME)
# MODEL_PATH = os.path.join(package_path, f"resource/{MODEL_NAME}")
MODEL_PATH = "/home/kiwi/github_package/rokeypj_ws/src/turtlebot3_autorace_detect/resource/멈처.tflite"


class WakeupWord:
    def __init__(self, buffer_size=2048):
        openwakeword.utils.download_models()
        self.buffer_size = buffer_size
        self.stream = None

        # 모델 로드
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.model_name = list(self.model.wakeword_models.keys())[0]
        print(f"Loaded model: {MODEL_PATH} (key: {self.model_name})")

        # 터미널 입력 설정
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        # 마이크 설정
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            input=True,
            frames_per_buffer=self.buffer_size,
        )

    def is_wakeup(self):
        # 엔터 키 입력 체크
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == '\n':
                print("[KEY] Enter pressed – wakeword accepted")
                return True

        # 오디오 수집 및 처리
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))

        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs.get(self.model_name, 0.0)
        print(f"[AUDIO] confidence: {confidence:.3f}")

        if confidence > 0.25:
            print("[AUDIO] Wakeword Detected!")
            return True
        return False

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def main():
    ww = WakeupWord()
    try:
        while not ww.is_wakeup():
            pass
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        ww.close()


if __name__ == '__main__':
    main()
