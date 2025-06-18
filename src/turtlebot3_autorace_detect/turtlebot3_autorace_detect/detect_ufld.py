#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

### ---- 통합된 backbone ----

class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4

### ---- 통합된 model ----

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='18', cls_dim=(101, 56, 4)):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim
        self.total_dim = np.prod(cls_dim)

        self.model = resnet(backbone, pretrained=pretrained)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512, 8, 1)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        fea = self.pool(fea).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        return group_cls

### ---- ROS2 Node ----

class UltraFastLaneDetector(Node):
    def __init__(self):
        super().__init__('detect_ufld_node')

        self.bridge = CvBridge()
        self.backbone = '18'
        self.griding_num = 101  # weight에 맞게 CULane 기준
        self.cls_num_per_lane = 56
        self.num_lanes = 4

        package_path = get_package_share_directory('turtlebot3_autorace_detect')
        model_path = os.path.join(package_path, 'model', 'tusimple_18.pth')  # weight 경로 (파일명만 tusimple이고 실제론 culane weight)

        self.net = parsingNet(size=(288, 800), backbone=self.backbone,
                               cls_dim=(self.griding_num, self.cls_num_per_lane, self.num_lanes))
        state_dict = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(state_dict['model'])
        self.net.eval()

        self.transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])

        self.sub_image = self.create_subscription(CompressedImage, '/camera/preprocessed/compressed', self.image_callback, 1)
        self.pub_image = self.create_publisher(CompressedImage, '/detect/image_output/compressed', 1)
        self.pub_center = self.create_publisher(Float64, '/detect/lane', 1)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        center_x, vis_img = self.process(cv_image)

        compressed_msg = self.bridge.cv2_to_compressed_imgmsg(vis_img, dst_format='jpg')
        self.pub_image.publish(compressed_msg)

        if center_x is not None:
            center_normalized = (center_x - 400) / 400.0  # -1.0 ~ +1.0 normalize
            self.pub_center.publish(Float64(data=float(center_normalized)))
            self.get_logger().info(f"Center X: {center_normalized}")

    def process(self, bgr_img):
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_input = self.transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            output = self.net(img_input)

        output = output.cpu().numpy()[0]  # shape: (4, 56, 101)
        out = np.argmax(output, axis=2)   # shape: (4, 56)

        col_sample = np.linspace(0, 800 - 1, self.griding_num)

        lanes_x = []
        for lane_idx in range(self.num_lanes):
            lane = out[lane_idx]
            lane_pos = []
            for i in range(self.cls_num_per_lane):
                if lane[i] < self.griding_num:
                    x = col_sample[lane[i]]
                    lane_pos.append(x)
                else:
                    lane_pos.append(-1)
            lanes_x.append(lane_pos)

        valid_lanes = [x for x in lanes_x if x[-1] >= 0]

        if len(valid_lanes) >= 2:
            left = valid_lanes[0][-1]
            right = valid_lanes[-1][-1]
            center_x = (left + right) / 2
        elif len(valid_lanes) == 1:
            center_x = valid_lanes[0][-1]
        else:
            center_x = None

        vis_img = self.visualize(bgr_img, lanes_x)
        return center_x, vis_img

    def visualize(self, img, lanes_x):
        vis = img.copy()
        h, w, _ = img.shape
        for lane in lanes_x:
            for idx, x in enumerate(lane):
                if x > 0:
                    y = int(h * (idx + 1) / self.cls_num_per_lane)
                    cv2.circle(vis, (int(x * w / 800), y), 3, (0, 255, 0), -1)
        return vis

def main(args=None):
    rclpy.init(args=args)
    node = UltraFastLaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
