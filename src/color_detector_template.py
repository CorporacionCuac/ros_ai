#!/usr/bin/env python3

import rospy #importar ros para python
from std_msgs.msg import String, Int32 # importar mensajes de ROS tipo String y tipo Int32
from sensor_msgs.msg import Image # importar mensajes de ROS tipo Image
import cv2 # importar libreria opencv
from cv_bridge import CvBridge # importar convertidor de formato de imagenes
import numpy as np # importar libreria numpy


class ColorDetector(object):
    def __init__(self):
        super(ColorDetector, self).__init__()
        self.pub = rospy.Publisher("topico_donde_publica", Image, queue_size=10)
        self.sub = rospy.Subscriber("topico_al_que_se_suscribe", Image, self.callback)

        self.bridge = CvBridge()


    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg)


    def procesar_img(self, img):

        # Cambiar espacio de color

        # Filtrar rango util

        # Aplicar mascara

        # Aplicar transformaciones morfologicas

        # Definir blobs

        # Dibujar rectangulos de cada blob

        # Publicar imagen final
        image_msg = self.bridge.cv2_to_imgmsg(cv_image)
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
