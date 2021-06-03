#!/usr/bin/env python3

import rospy #importar ros para python
from sensor_msgs.msg import Image, CompressedImage # importar mensajes de ROS tipo Image
import cv2 # importar libreria opencv
from cv_bridge import CvBridge # importar convertidor de formato de imagenes
import numpy as np # importar libreria numpy
import time


class ColorDetector(object):
    def __init__(self):
        super(ColorDetector, self).__init__()
        self.pub = rospy.Publisher("/cuac/images/pato", Image, queue_size=1)
        self.sub = rospy.Subscriber("/cuac/corrected_image/compressed", CompressedImage, self.callback)

        self.bridge = CvBridge()

        ### Resuelve duda planteada por Efrain
        #self.sync()

    def sync(self):
        while True:
            msg = rospy.wait_for_message("/cuac/corrected_image/compressed", CompressedImage)
            self.callback(msg)

    def callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.procesar_img(cv_image)

    def procesar_img(self, img):
        # Cambiar espacio de color

        # Filtrar rango util

        # Aplicar mascara

        # Aplicar transformaciones morfologicas

        # Definir blobs

        # Dibujar rectangulos de cada blob

        # Publicar imagen final
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub.publish(image_msg)

def main():
    rospy.init_node('ColorDetector') #creacion y registro del nodo!

    try:
        obj = ColorDetector() # Crea un objeto del tipo ColorDetector, cuya definicion se encuentra arriba
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo('Closing ColorDetector')


if __name__ =='__main__':
    main()
