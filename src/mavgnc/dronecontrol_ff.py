import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from scipy.spatial.transform import Rotation as R
from tf.transformations import quaternion_from_euler,euler_from_quaternion

class DroneControl_feed_foward:
    def loaddata(self,path):
        data = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=2) 
        q = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,3],data[:,8],data[:,9],data[:,10],data[:,5],data[:,6],data[:,7],data[:,4],data[:,20],data[:,21],data[:,22],data[:,23],data[:,11],data[:,12],data[:,13]))
        return q

    def time_search(self, time, t_list):
        if self.pointer < len(t_list)-1:
            if (time >= t_list[self.pointer] and time <t_list[self.pointer+1]):  
                return self.pointer
            elif time > t_list[self.pointer + 1]:
                self.pointer += 1
        return self.pointer 


    def __init__(self,file_path):
        self.k = 0
        self.path = file_path
        self.plan_cmd = self.loaddata(self.path)
        self.pointer = 0

        self.forward_thrust_cmd = 0
        self.forward_bodyrate_cmd = np.array((3,))
        self.forward_orien_cmd = np.array((4,))
        self.forward_attitude_cmd = np.array((3,))

        self.R_ENU2NED = R.from_euler('zyx', [0,0,180], degrees=True)
        

    def set_forwardcontrol(self,t):
        
        self.k = self.time_search(t,self.plan_cmd[:,0])
        # print(self.k)
        if self.k < self.plan_cmd.shape[0] - 1:
            self.forward_control(self.plan_cmd[self.k,:])
            is_done = True
            return self.forward_position_cmd,self.forward_velocity_cmd,self.forward_thrust_cmd,self.forward_attitude_cmd,self.forward_bodyrate_cmd,is_done
            # self.planning = self.plan_att[self.k,1:6]
        else:
            self.forward_control(self.plan_cmd[self.plan_cmd.shape[0] - 1,:])
            is_done = False
            return self.forward_position_cmd,self.forward_velocity_cmd,self.forward_thrust_cmd,self.forward_attitude_cmd,self.forward_bodyrate_cmd,is_done

    def forward_control(self,data): 
        self.forward_position_cmd = np.array([data[1],data[2],data[3]])
        # print(self.forward_position_cmd)
        self.forward_velocity_cmd = np.array([data[4],data[5],data[6]])
        self.forward_thrust_cmd = data[11] + data[12] + data[13] + data[14]
        # print(self.forward_thrust_cmd)
        self.forward_orien_cmd = np.array([data[7],data[8],data[9],data[10]])
        # print(self.forward_orien_cmd)
        self.forward_bodyrate_cmd = np.array([data[15],data[16],data[17]])
        self.forward_attitude_cmd = np.array(euler_from_quaternion([self.forward_orien_cmd[0],self.forward_orien_cmd[1],self.forward_orien_cmd[2],self.forward_orien_cmd[3]]))
        # print(self.forward_attitude_cmd)
        

if __name__ == "__main__":
    dronecontol_ff = DroneControl_feed_foward("/home/zhoujin/rainsunny_ws/src/Mavfast/src/dataset/8_shape1.csv")
    a = dronecontol_ff.set_forwardcontrol(1)
    print(a)
