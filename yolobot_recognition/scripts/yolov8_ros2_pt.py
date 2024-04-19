#!/usr/bin/env python3
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


from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()
height = 640 
width = 640
xcr = 0
ycr = 0
thetacr = 0
pcr = np.array([[xcr], [ycr], [thetacr]]) #current posture (xc, yc, thetac) coordenadas globais do eixo do centro do eixo do robô
xcc = 0
ycc = 392.5 #mm 
zcc = 352.5 #mm
pcc = np.array([[xcc], [ycc], [zcc]]) #current posture (xc, yc, zcc) coordenadas da camera em relacao ao eixo do centro do eixo do robô
p = 20
v = 200 #módulo velocidade do robô em mm/s
i = 0 #numero de iteracoes 
k = 3.7795275590551 #conversao de mm para pixel
cbe = 0 
cbd = 0
conta_boxes_e = []
conta_boxes_d = []
x_e = 0
x_d = 0
lista_y_e = np.array([[]])
lista_y_d = np.array([[]])
lista_x_e = np.array([[]])
lista_x_d = np.array([[]])
lista_z_e = np.array([[]])
lista_z_d = np.array([[]])

    
class Camera_subscriber(Node):  

    def __init__(self):
        super().__init__('camera_subscriber')

        self.model = YOLO('~/yolobot/src/yolobot_recognition/scripts/modelo.pt')

        self.yolov8_inference = Yolov8Inference()

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)
        self.subscription 

        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.erro_pub = self.create_publisher(Float64, '/erro', 1)


    def camera_callback(self, data):

        global width 
        global height 
        global xcr 
        global ycr 
        global thetacr 
        global pcr 
        global xcc 
        global ycc 
        global zcc 
        global pcc
        global v
        global p
        global i  
        global k 
        global cbe
        global cbd
        global conta_boxes_e
        global conta_boxes_d 
        global x_e
        global x_d
        global lista_y_e
        global lista_y_d 
        global lista_x_e
        global lista_x_d 
        global lista_z_e 
        global lista_z_d

        img = bridge.imgmsg_to_cv2(data)
        results = self.model(img)

        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = camera_subscriber.get_clock().now().to_msg()

        for r in results:
            boxes = r.boxes
            t = (r.speed['inference'])/1000 #tempo em s
            # boxes = r.boxes.xywh.cpu().numpy() #todas as boxes de um frame

            for box in boxes:
                self.inference_result = InferenceResult()
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                self.inference_result.class_name = self.model.names[int(c)]
                self.inference_result.top = int(b[0])
                self.inference_result.left = int(b[1])
                self.inference_result.bottom = int(b[2])
                self.inference_result.right = int(b[3])
                self.yolov8_inference.yolov8_inference.append(self.inference_result)

                box_xc = (int(b[2])+ int(b[0]))/2
                box_zc = (int(b[3])+ int(b[1]))/2

                if box_xc < width/2:
                    if box_zc > height*2/3:
                        lista_x_e = np.append(lista_x_e, [[(xcr + ycc*np.sin(thetacr) + ((box_xc-width/2)*np.cos(thetacr))/k)]], axis=1) #lista de x das boxes da iteraçao em coordenadas globais
                        lista_z_e = np.append(lista_z_e, [[(height/2 - box_zc)/k + zcc]], 1)
                        lista_y_e = np.append(lista_y_e, [[ycr + ycc*np.cos(thetacr) + ((box_xc-width/2)*np.sin(thetacr))/k]], 1)
                        cbe += 1
                else:
                    if box_zc > height*2/3:
                        lista_x_d = np.append(lista_x_d, [[(xcr + ycc*np.sin(thetacr) + ((box_xc-width/2)*np.cos(thetacr))/k)]], 1)
                        lista_z_d = np.append(lista_z_d, [[(height/2 - box_zc)/k + zcc]], 1)
                        lista_y_d = np.append(lista_y_d, [[ycr + ycc*np.cos(thetacr) + ((box_xc-width/2)*np.sin(thetacr))/k]], 1)
                        cbd += 1

                # if box_xc < width/2:
                #     if cbe == 0:
                #         aux1 = xcr + ycc*np.sin(thetacr) + ((box_xc-width/2)*np.cos(thetacr))/k
                #         aux2 = (height/2 - box_zc)/k + zcc
                #         aux3 = ycr + ycc*np.cos(thetacr) + ((box_xc-width/2)*np.sin(thetacr))/k
                #         lista_x_e = np.append(lista_x_e, [[aux1]], axis=1) #lista de x das boxes da iteraçao em coordenadas globais
                #         lista_z_e = np.append(lista_z_e, [[aux2]], 1)
                #         lista_y_e = np.append(lista_y_e, [[aux3]], 1)
                #     else:
                #         lista_x_e = np.append(lista_x_e, [[aux1]], axis=1) #lista de x das boxes da iteraçao em coordenadas globais
                #         lista_z_e = np.append(lista_z_e, [[aux2]], 1)
                #         lista_y_e = np.append(lista_y_e, [[aux3 + p]], 1)
                #         aux3 += p
                #     cbe += 1
                # else:
                #     if cbd == 0:
                #         aux4 = xcr + ycc*np.sin(thetacr) + ((box_xc-width/2)*np.cos(thetacr))/k
                #         aux5 = (height/2 - box_zc)/k + zcc
                #         aux6 = ycr + ycc*np.cos(thetacr) + ((box_xc-width/2)*np.sin(thetacr))/k
                #         lista_x_d = np.append(lista_x_d, [[aux4]], 1)
                #         lista_z_d = np.append(lista_z_d, [[aux5]], 1)
                #         lista_y_d = np.append(lista_y_d, [[aux6]], 1)
                #     else:
                #         lista_x_d = np.append(lista_x_d, [[aux4]], 1)
                #         lista_z_d = np.append(lista_z_d, [[aux5]], 1)
                #         lista_y_d = np.append(lista_y_d, [[aux6 + p]], 1)
                #         aux6 += p
                #     cbd += 1
        
        conta_boxes_e.append(cbe)
        conta_boxes_d.append(cbd)
        cbe, cbd = 0, 0

        if i > 7: #quando tiver 10 interaçoes começa-se a colocar o mais novo e tirar o mais antigo
            lista_y_e = np.delete(lista_y_e, np.s_[0:conta_boxes_e[0]], 1)
            lista_x_e = np.delete(lista_x_e, np.s_[0:conta_boxes_e[0]], 1)
            lista_z_e = np.delete(lista_z_e, np.s_[0:conta_boxes_e[0]], 1)
            lista_y_d = np.delete(lista_y_d, np.s_[0:conta_boxes_d[0]], 1)
            lista_x_d = np.delete(lista_x_d, np.s_[0:conta_boxes_d[0]], 1)
            lista_z_d = np.delete(lista_z_d, np.s_[0:conta_boxes_d[0]], 1)
            conta_boxes_e = conta_boxes_e[1:len(conta_boxes_e)]
            conta_boxes_d = conta_boxes_d[1:len(conta_boxes_d)]

        if lista_x_e.size > 1:
            auxe = np.concatenate((lista_y_e.T, lista_z_e.T), axis=1)
            VIe = pd.DataFrame(auxe)
            VDe = pd.DataFrame(lista_x_e.T)
            reg_e = linear_model.LinearRegression()
            reg_e.fit(VIe, VDe)
            x_e = (reg_e.predict([[ycr + ycc*np.cos(thetacr) - (width/4)*np.sin(thetacr)/k, zcc]])[0][0])*k

        if lista_x_d.size > 1:
            auxd = np.concatenate((lista_y_d.T, lista_z_d.T), axis=1)
            VId = pd.DataFrame(auxd)
            VDd = pd.DataFrame(lista_x_d.T)
            reg_d = linear_model.LinearRegression()
            reg_d.fit(VId, VDd)
            x_d = (reg_d.predict([[ycr + ycc*np.cos(thetacr) + (width/4)*np.sin(thetacr)/k, zcc]])[0][0])*k

        if lista_x_d.size == 0:
            x_d = (width/2)
        if lista_x_e.size == 0:
            x_e = -(width/2)

        i += 1
        xrr = (x_e + x_d)/2
        erro = Float64()
        erro.data = 2*xrr/width

        self.erro_pub.publish(erro) #publica erro

        ycr += v*t
        c = int(xrr + width/2)
        annotated_frame = results[0].plot()   
        imagem = cv2.line(annotated_frame, (c, int(height/2)), (c, int(height)), (0, 0, 255), 2) # vermelho
        robo = cv2.line(imagem, (width//2, int(height/2)), (width//2, int(height)), (0, 255, 255), 2) # vermelho

            #camera_subscriber.get_logger().info(f"{self.yolov8_inference}")

        img_msg = bridge.cv2_to_imgmsg(robo)  

        self.img_pub.publish(img_msg)
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

if __name__ == '__main__':
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
