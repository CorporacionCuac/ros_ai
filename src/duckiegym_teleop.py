#!/usr/bin/env python3

import rospy

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from duckiegym_ros.msg import Twist2DStamped


class Mapping:
    def __init__(self):
        # publicador
        self.pub_sim = rospy.Publisher("/cuac/lane_controller_node/car_cmd", Twist2DStamped, queue_size=1)

        # subscriptor
        self.sub = rospy.Subscriber("/keys", String, self.callback)


    def callback(self, msg):
        rospy.loginfo(msg)
        # logica de tomar el mensaje y publicar a la tortuga
        key = msg.data

        t = Twist2DStamped()
        if key == 's':
            pass
        elif key == 'w':
            t.v = 0.5
        elif key == 'x':
            t.v = -0.5
        elif key == 'a':
            t.omega = 0.5
        elif key == 'd':
            t.omega = -0.5
        elif key == 'q':
            t.v = 0.5
            t.omega = 0.5
        elif key == 'e':
            t.v = 0.5
            t.omega = -0.5

        self.pub_sim.publish(t)



if __name__ == '__main__':
    rospy.init_node('duckiegym_teleop')
    rospy.loginfo('Init Mapping')
    try:
        Mapping()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo('Closing Mapping')
