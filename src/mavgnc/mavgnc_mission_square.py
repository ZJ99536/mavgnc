from mavgnc.mavgnc_mission_base import MavGNCMissionPosBase
import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

class MavGNCMissionSquare(MavGNCMissionPosBase):
    def __init__(self):
        super(MavGNCMissionSquare,self).__init__()
        self.home = (self.local_position.pose.position.x,self.local_position.pose.position.y,self.local_position.pose.position.z)
        self.waypoints = np.array([[self.home[0],self.home[1],self.home[2]+30],
            [self.home[0]+500,self.home[1],self.home[2]+30],
            [self.home[0]+500,self.home[1]+500,self.home[2]+30],
            [self.home[0],self.home[1]+500,self.home[2]+30],
            [self.home[0],self.home[1],self.home[2]+30]])
        self.waypoint_counter = 0

    def mission(self):
        self.pos.pose.position.x = self.waypoints[self.waypoint_counter,0] 
        self.pos.pose.position.y = self.waypoints[self.waypoint_counter,1] 
        self.pos.pose.position.z = self.waypoints[self.waypoint_counter,2] 

        # For demo purposes we will lock yaw/heading to north.
        yaw_degrees = 0  # North
        yaw = math.radians(yaw_degrees)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

        if self.is_at_position(self.pos.pose.position.x,self.pos.pose.position.y,self.pos.pose.position.z,0.5):
            self.waypoint_counter += 1 
            if self.waypoint_counter == self.waypoints.shape[0]:
                self.waypoint_counter = 0


if __name__ == '__main__':
    rospy.init_node('squre', anonymous=True)
    mission_square = MavGNCMissionSquare()
    mission_square.run()

