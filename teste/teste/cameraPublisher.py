import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from sklearn import linear_model
import pandas as pd
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.cameraDeviceNumber = 0
        self.camera = cv2.VideoCapture(0)
        self.bridgeObject = CvBridge()
        self.queueSize = 20
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw',self.queueSize)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
    

    def timer_callback(self):
        success, frame = self.camera.read()
        frame = cv2.resize(frame, (640,640),interpolation=cv2.INTER_CUBIC)
        if success == True:
            self.publisher_.publish(self.bridgeObject.cv2_to_imgmsg(frame))
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisherObject = MinimalPublisher()
    rclpy.spin(publisherObject)
    publisherObject.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()