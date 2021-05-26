#!/usr/bin/env python3
 
import rospy

from std_msgs.msg import String
from geometry_msgs.msg import Twist


class Mapping:
    def __init__(self):
        # publicador
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)

        # subscriptor
        self.sub = rospy.Subscriber("/keys", String, self.callback)
        

    def callback(self, msg):
        rospy.loginfo(msg)
        # logica de tomar el mensaje y publicar a la tortuga

        t = Twist()

        self.pub.publish(t)



if __name__ == '__main__':
    rospy.init_node('turtle_teleop')
    rospy.loginfo('Init Mapping')
    try:
        Mapping()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo('Closing Mapping')