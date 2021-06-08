#!/usr/bin/env python3

import rospy

from std_msgs.msg import String
from duckiegym_ros.msg import Twist2DStamped


class Mapping:
    def __init__(self):
        # publicador
        self.pub_sim = rospy.Publisher("/cuac/lane_controller_node/car_cmd", Twist2DStamped, queue_size=1)

        # subscriptor
        self.sub = rospy.Subscriber("/emotions", String, self.callback)


    def callback(self, msg):
        rospy.loginfo(msg)
        # logica de tomar el mensaje y publicar al duckiebot
        key = msg.data

        t = Twist2DStamped()
        if key == 'Neutral':
            pass
        elif key == 'Happy':
            t.v = 0.5
        elif key == 'Sad':
            t.omega = 0.5
        elif key == 'Surprise':
            t.omega = -0.5
        elif key == 'Angry':
            t.v = 1

        self.pub_sim.publish(t)



if __name__ == '__main__':
    rospy.init_node('duckiegym_face_teleop_node')
    rospy.loginfo('Init Mapping')
    try:
        Mapping()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo('Closing Mapping')
