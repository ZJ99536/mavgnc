#!/usr/bin/env python2
#***************************************************************************
#
#   Copyright (c) 2015 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#***************************************************************************/

#
# @author Andreas Antener <andreas@uaventure.com>
#
# The shebang of this file is currently Python2 because some
# dependencies such as pymavlink don't play well with Python3 yet.
from __future__ import division

from matplotlib.pyplot import flag

import rospy
import math
from math import sin,cos,tan,atan,asin,acos,atan2
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3,TwistStamped,Vector3Stamped
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float32 
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header
from threading import Thread
from tf.transformations import quaternion_from_euler,euler_from_quaternion,quaternion_from_matrix
from mavros_msgs.msg import AttitudeTarget
from mavgnc.mavgnc_base import MavGNCBase
from time import time
import geometry_msgs
from mavgnc.dronecontrol_ff import DroneControl_feed_foward


class MavGNCPositionControl(MavGNCBase):
    """
    Tests flying a path in offboard control by sending position setpoints
    via MAVROS.

    For the test to be successful it needs to reach all setpoints in a certain time.

    FIXME: add flight path assertion (needs transformation from ROS frame to NED)
    """
    def __init__(self):
        super(MavGNCPositionControl,self).__init__()
        self.ready_to_takeoff = False
        self.mission_finish = False
        self.mission_ready= False
        self.att = AttitudeTarget() 
        self.att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        self.vel_setpoint_pub = rospy.Publisher('mavgnc/velocity_setpoint', TwistStamped, queue_size=1)
        self.pos_setpoint_pub = rospy.Publisher('mavgnc/position_setpoint', PoseStamped, queue_size=1)
        self.time_pub = rospy.Publisher('mavgnc/time', Float32, queue_size=1)
        self.odom_sub = rospy.Subscriber('mavros/local_position/odom',Odometry,self.odometry_cb)
        self.att_sp_euler_pub = rospy.Publisher('mavgnc/att_sp_euler',Vector3Stamped,queue_size=1)
        self.att_euler_pub = rospy.Publisher('mavgnc/att_euler',Vector3Stamped,queue_size=1)
        

        self.position_setpoint = PoseStamped() 
        self.velocity_setpoint = TwistStamped() 
        self.att_setpoint_euler = Vector3Stamped()
        self.attitude_euler = Vector3Stamped()

        # self.ff_controller = DroneControl_feed_foward("/home/zhoujin/rainsunny_ws/src/Mavfast/src/dataset/8_shape1.csv")
        self.ff_controller = DroneControl_feed_foward("/home/zhoujin/Mavfast/src/dataset/8_shape_planning1.csv")

        self.current_position = np.array((3,))
        self.current_velocity= np.array((3,))
        self.current_attitude = np.array((3,))

        self.hp = np.array((3,))
        self.status = 'Hover'
        self.flag = 0

        self.pos_ff = 0
        self.vel_ff = 0
        self.thrust_ff = 0
        self.att_ff = 0
        self.rate_ff = 0

        self.phi_cmd = 0.0
        self.theta_cmd = 0.0
        self.psi_cmd = 0.0
        self.thrust_cmd = 0.0

        self.kp_x = 0.7
        self.kp_y = 0.7 
        self.kp_z = 0.7 

        self.kp_vx = 0.3
        self.kp_vy = -0.3
        self.kp_vz = 0.5

        self.ki_x = 0
        self.ki_y = 0
        self.ki_z = 0.1

        self.kd_x = 0.01
        self.kd_y = 0.015
        self.kd_z = 0.015

        self.ki_vx = 0.1
        self.ki_vy = 0.15
        self.ki_vz = 0.2

        self.kd_vx = 0.1
        self.kd_vy = 0.15
        self.kd_vz = 0.1

        self.vel_err_sum = np.zeros(3)
        self.vel_err_last_step = np.zeros(3)
        self.pos_err_sum = np.zeros(3)
        self.pos_err_last_step = np.zeros(3)

 
        self.ts = 0
        self.tss = None
        self.tsa = None
        self.n_seg = 0
        self.n_order = 7
        self.Q = None
        self.M = None
        self.C = None
        self.Rp = None
        self.Rpp = None
        self.Rfp = None
        self.polyx = None
        self.polyy = None
        self.polyz = None
        self.tempi = 0
        self.endx = 0
        self.endy = 0
        self.endz = 0

        self.cut_seg = 6
        self.eight_turns = 2
        self.eight_ax = 10
        self.eight_ay = 15
        self.eight_t = 2*np.pi/self.cut_seg
        self.start_t = 0
        self.current_t = 0

        self.g = 9.81
        self.k_p_fb = 1.3
        self.k_v_fb = 3.5
        self.k_p_att_euler = [5, 5, 5]

        self.vxmax = 3.5
        self.vymax = 3.5
        self.vzmax = 3.5


        self.att_thread = Thread(target=self.send_att, args=())
        self.att_thread.daemon = True
        self.att_thread.start()
        self.loop_freq = 200
        self.loop_rate = rospy.Rate(self.loop_freq)

        self.time_init = time()
        self.current_time = Float32()
        self.current_time.data = .0




    def odometry_cb(self,data):
        self.current_position = np.array([data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z])
        self.current_velocity = np.array([data.twist.twist.linear.x,data.twist.twist.linear.y,data.twist.twist.linear.z])
        self.current_attitude = np.array(euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w]))
        
        
    #
    # Helper methods
    #
    def send_att(self):
        rate = rospy.Rate(200)  # Hz
        self.att.header = Header()
        self.att.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.att_setpoint_pub.publish(self.att)
            self.vel_setpoint_pub.publish(self.velocity_setpoint)
            self.pos_setpoint_pub.publish(self.position_setpoint)
            self.att_euler_pub.publish(self.attitude_euler)
            self.att_sp_euler_pub.publish(self.att_setpoint_euler)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        pass


    def run(self):
        while True:            
            if not self.ready_to_takeoff:
                self.takeoff_preparation()
            else:
                if self.status == 'Hover':
                    self.is_at_setpoint()
                    self.position_control_fb()
                    self.velocity_control_fb()
                    self.start_t = time()
                else:
                    self.current_t = time()-self.start_t
                    self.current_time.data = self.current_t
                    self.planner()
                    self.position_control()
                    self.velocity_control()
            self.time_pub.publish(self.current_time)
            self.loop_rate.sleep()

    def is_at_setpoint(self):
        self.position_setpoint.pose.position.x = 0
        self.position_setpoint.pose.position.y = 0
        self.position_setpoint.pose.position.z = 2   

        self.hp = np.array([self.position_setpoint.pose.position.x,self.position_setpoint.pose.position.y,self.position_setpoint.pose.position.z])
        dis = self.current_position - self.hp
        
        if np.linalg.norm(dis) < 0.1:
            self.status = 'Planning'
        else:
            self.status = 'Hover'

    def position_control_fb(self):
        position_cmd = np.array([self.position_setpoint.pose.position.x,self.position_setpoint.pose.position.y,self.position_setpoint.pose.position.z])
        pos_err = position_cmd - self.current_position

        self.velocity_setpoint.twist.linear.x = self.kp_x * pos_err[0]
        self.velocity_setpoint.twist.linear.y = self.kp_y * pos_err[1]
        self.velocity_setpoint.twist.linear.z = self.kp_z * pos_err[2]

        self.position_setpoint.header.stamp = rospy.Time.now() 
        self.position_setpoint.header.frame_id = 'odom'
        self.velocity_setpoint.header.stamp = rospy.Time.now()  
        self.velocity_setpoint.header.frame_id = 'odom'

        self.psi_cmd = 0.0

    def velocity_control_fb(self):
        velocity_cmd = np.array([self.velocity_setpoint.twist.linear.x,self.velocity_setpoint.twist.linear.y,self.velocity_setpoint.twist.linear.z])
        vel_err = velocity_cmd - self.current_velocity

        psi = self.current_attitude[2]
        
        R_E_B = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        vel_err = R_E_B@vel_err

        self.vel_err_sum += vel_err * 1.0/self.loop_freq
        
        self.thrust_cmd = 0.68 + self.kp_vz * vel_err[2] + self.ki_vz * self.vel_err_sum[2] + self.kd_vz * (vel_err[2] - self.vel_err_last_step[2])*self.loop_freq
        if self.thrust_cmd >= 1:
            self.thrust_cmd = 0.99
        elif self.thrust_cmd <= 0:
            self.thrust_cmd = 0.01

        self.theta_cmd = self.kp_vx * vel_err[0]
        self.phi_cmd = self.kp_vy * vel_err[1]
        
        self.att.orientation = Quaternion(*quaternion_from_euler(self.phi_cmd,self.theta_cmd,self.psi_cmd))
        self.att.thrust = self.thrust_cmd
        self.att.header.stamp = rospy.Time.now()
        self.att.body_rate = Vector3()
        self.att.type_mask = 7 # ignore rate

        self.att_setpoint_euler.vector.x = self.phi_cmd/3.14*180
        self.att_setpoint_euler.vector.y = self.theta_cmd/3.14*180
        self.att_setpoint_euler.vector.z = self.psi_cmd/3.14*180
        self.att_setpoint_euler.header.stamp = rospy.Time.now()

        self.attitude_euler.vector.x = self.current_attitude[0]/3.14*180
        self.attitude_euler.vector.y = self.current_attitude[1]/3.14*180
        self.attitude_euler.vector.z = self.current_attitude[2]/3.14*180
        self.attitude_euler.header.stamp = rospy.Time.now()

        self.vel_err_last_step = vel_err

    def takeoff_preparation(self):
        # make sure the simulation is ready to start the mission
        self.wait_for_topics(60)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 10, -1)
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)
        self.ready_to_takeoff = True
        return True

    def velocity_control(self):
        position_cmd = np.array([self.position_setpoint.pose.position.x,self.position_setpoint.pose.position.y,self.position_setpoint.pose.position.z])
        position_cmd = self.pos_ff
        pos_err = position_cmd - self.current_position
        self.pos_err_sum += pos_err * 1.0/self.loop_freq
        velocity_cmd = np.array([self.velocity_setpoint.twist.linear.x,self.velocity_setpoint.twist.linear.y,self.velocity_setpoint.twist.linear.z])
        position_cmd = self.vel_ff
        vel_err = velocity_cmd - self.current_velocity
        self.vel_err_sum += vel_err * 1.0/self.loop_freq

        phi = self.current_attitude[0]
        theta = self.current_attitude[1]
        psi = self.current_attitude[2]
        R_E_B = np.array([[cos(psi),sin(psi),0],[-sin(psi),cos(psi),0],[0,0,1]])
        vel_err = R_E_B@vel_err

        phi = self.att_ff[0]
        theta = self.att_ff[1]
        psi = self.att_ff[2]
        K_pos = np.array([[self.k_p_fb,0,0],[0,self.k_p_fb,0],[0,0,self.k_p_fb]])
        K_vel = np.array([[2,0,0],[0,2,0],[0,0,2]])
        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])
        # print(self.thrust_ff)
        if self.thrust_ff < 0.01:
            self.thrust_ff = self.g
        # print(self.thrust_ff)
        # print(self.pos_ff)
        aref = R_E_B.transpose() @ np.array([0.0,0.0,self.thrust_ff]) - self.g*np.array([0,0,1])
        ades = aref + self.g*np.array([0,0,1]) + K_pos @ pos_err + K_vel @ vel_err                
        acc_des = ades
        # print(acc_des)
        acc_des[0] += self.ki_vx * self.vel_err_sum[0] + self.kd_vx * (vel_err[0] - self.vel_err_last_step[0])*self.loop_freq
        acc_des[0] += self.ki_x * self.pos_err_sum[0] + self.kd_x * (pos_err[0] - self.pos_err_last_step[0])*self.loop_freq
        acc_des[1] += self.ki_vy * self.vel_err_sum[1] + self.kd_vy * (vel_err[1] - self.vel_err_last_step[1])*self.loop_freq
        acc_des[1] += self.ki_y * self.pos_err_sum[1] + self.kd_y * (pos_err[1] - self.pos_err_last_step[1])*self.loop_freq
        acc_des[2] += self.ki_vz * self.vel_err_sum[2] + self.kd_vz * (vel_err[2] - self.vel_err_last_step[2])*self.loop_freq
        acc_des[2] += self.ki_z * self.pos_err_sum[2] + self.kd_z * (pos_err[2] - self.pos_err_last_step[2])*self.loop_freq
        if acc_des[2] < 0.01:
            acc_des[2] = 0.01
        z_b_des = np.array(acc_des / np.linalg.norm(acc_des))
        y_c = np.array([-sin(psi),cos(psi),0])
        x_b_des = np.cross(y_c,z_b_des) / np.linalg.norm(np.cross(y_c,z_b_des))
        y_b_des = np.cross(z_b_des,x_b_des)
        R_E_B = np.transpose(np.array([x_b_des,y_b_des,z_b_des]))
        self.psi_cmd = atan2(R_E_B[1,0],R_E_B[0,0])
        self.theta_cmd = asin(-R_E_B[2,0])
        self.phi_cmd = atan(R_E_B[2,1]/R_E_B[2,2])
        self.thrust_cmd = np.linalg.norm(acc_des)*0.68/self.g

        self.theta_cmd = self.bound(self.theta_cmd,-0.55,0.55)
        self.phi_cmd = self.bound(self.phi_cmd,-0.62,0.62)
        self.thrust_cmd = self.bound(self.thrust_cmd,0,0.95)

        # if abs(self.psi_cmd) > 1:
        #     print(acc_des)

        att_cmd = np.array([self.phi_cmd,self.theta_cmd,self.psi_cmd])
        w_fb = self.k_p_att_euler * (att_cmd - self.current_attitude)
        w_cmd = self.rate_ff + w_fb
        self.att.orientation = Quaternion(*quaternion_from_euler(self.phi_cmd,self.theta_cmd,self.psi_cmd))
        self.att.thrust = self.thrust_cmd
        self.att.body_rate.x = w_cmd[0]
        self.att.body_rate.y = w_cmd[1]
        self.att.body_rate.z = w_cmd[2]
        self.att.type_mask = 128 # ignore rate
        self.att.header.stamp = rospy.Time.now()
        

        self.att_setpoint_euler.vector.x = self.phi_cmd/3.14*180
        self.att_setpoint_euler.vector.y = self.theta_cmd/3.14*180
        self.att_setpoint_euler.vector.z = self.psi_cmd/3.14*180
        self.att_setpoint_euler.header.stamp = rospy.Time.now()

        self.attitude_euler.vector.x = self.current_attitude[0]/3.14*180
        self.attitude_euler.vector.y = self.current_attitude[1]/3.14*180
        self.attitude_euler.vector.z = self.current_attitude[2]/3.14*180
        self.attitude_euler.header.stamp = rospy.Time.now()

        self.vel_err_last_step = vel_err



    def position_control(self):
        self.velocity_setpoint.twist.linear.x = self.vel_ff[0]
        self.velocity_setpoint.twist.linear.y = self.vel_ff[1]
        self.velocity_setpoint.twist.linear.z = self.vel_ff[2]
        self.velocity_setpoint.header.stamp = rospy.Time.now()  
        self.velocity_setpoint.header.frame_id = 'odom'

        self.psi_cmd = 0.0

    def bound(self,data,min_value,max_value):
        if data >=max_value:
            data = max_value
        elif data <=min_value:
            data = min_value
        return data

    def planner(self):
        self.pos_ff,self.vel_ff,self.thrust_ff,self.att_ff,self.rate_ff,is_done = self.ff_controller.set_forwardcontrol(self.current_t)
        
        self.position_setpoint.pose.position.x = self.pos_ff[0]
        self.position_setpoint.pose.position.y = self.pos_ff[1]
        self.position_setpoint.pose.position.z = self.pos_ff[2]
        self.position_setpoint.header.stamp = rospy.Time.now() 
        self.position_setpoint.header.frame_id = 'odom'




