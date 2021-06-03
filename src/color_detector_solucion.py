#!/usr/bin/env python3

import rospy #importar ros para python
from std_msgs.msg import String, Int32 # importar mensajes de ROS tipo String y tipo Int32
from sensor_msgs.msg import Image, CompressedImage # importar mensajes de ROS tipo Image
import cv2 # importar libreria opencv
from cv_bridge import CvBridge # importar convertidor de formato de imagenes
import numpy as np # importar libreria numpy


class ColorDetector(object):
    def __init__(self):
        super(ColorDetector, self).__init__()
        self.pub = rospy.Publisher("/cuac/images/pato", Image, queue_size=1)
        self.sub = rospy.Subscriber("/cuac/corrected_image/compressed", CompressedImage, self.callback)

        self.bridge = CvBridge()


    def callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.procesar_img(cv_image)


    def procesar_img(self, img):

        # Cambiar espacio de color
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Filtrar rango util
        lower_yellow = np.array([20, 200, 130])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        # Aplicar mascara
        img_out = cv2.bitwise_and(img, img, mask=mask)

        # Aplicar transformaciones morfologicas
        kernel = np.ones((3,3),np.uint8)

        mask = cv2.erode(mask, kernel, iterations = 1)
        mask = cv2.dilate(mask, kernel, iterations = 1)

        # Aplicar mascara
        img_out_m = cv2.bitwise_and(img, img, mask=mask)

        # Definir blobs
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # Dibujar rectangulos de cada blob
        for cnt in contours:
            # Obtener rectangulo que bordea un contorno
            x,y,w,h = cv2.boundingRect(cnt)
            #Filtrar por area minima
            if w*h > 1000:
                x2=x+w
                y2=y+h
                #Dibujar rectangulo en el frame original
                cv2.rectangle(img, (x,y), (x2,y2), (250,0,0), 2)
        # Publicar imagen final
        cv2.imshow('Filtrado', img_out)
        cv2.imshow('Morpho', img_out_m)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', img)
        cv2.waitKey(1)
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub.publish(image_msg)


def main():
    rospy.init_node('ColorDetector') #creacion y registro del nodo!

    try:
        ColorDetector() # Crea un objeto del tipo ColorDetector, cuya definicion se encuentra arriba
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo('Closing ColorDetector')


if __name__ =='__main__':
    main()
