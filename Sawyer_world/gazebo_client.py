#!/usr/bin/env python
#This is a sample code to control gazebo models using ros service commands
import sys
import rospy
import os
import math
import time

#most of the time t_eff will be negative, so it will keep applying effort forever, t_seq supposed to be halt time between two different joint actions
class ComClient(object):
    def __init__(self):
        self.D=[]
        self.E=[]
        self.R=[]
        self.S=[]
        self.call_t=time.time()
        self.exec_t=time.time()

    def _stack_actions(self,e,t_eff,t_seq,robot_name):
        self.E.reverse()
        self.D.reverse()
        self.R.reverse()
        self.S.reverse()
        D=[]
        E=[]
        S=[]
        for i in range(len(e)):
            D.append(repr(1000000000*t_eff[i]))
            E.append(repr(e[i]))
            S.append(repr(t_seq[i]))
        self.D.append(D)
        self.E.append(E)
        self.R.append(robot_name)
        self.S.append(S)

        self.E.reverse()
        self.D.reverse()
        self.R.reverse()
        self.S.reverse()

    def make_services(self,robot_name,effort,duration):
        self.s0= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+", effort: "+effort[0]+", start_time: 0, duration: "+duration[0]+"}'"
        self.s1= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j1"+"\""+", effort: "+effort[1]+", start_time: 0, duration: "+duration[0]+"}'"
        self.s2= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j2"+"\""+", effort: "+effort[2]+", start_time: 0, duration: "+duration[0]+"}'"
        self.s3= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j3"+"\""+", effort: "+effort[3]+", start_time: 0, duration: "+duration[0]+"}'"
        self.s4= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j4"+"\""+", effort: "+effort[4]+", start_time: 0, duration: "+duration[0]+"}'"
        self.s5= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j5"+"\""+", effort: "+effort[5]+", start_time: 0, duration: "+duration[0]+"}'"
        self.s6= "rosservice call gazebo/apply_joint_effort '{joint_name: "+"\""+robot_name+"::right_j6"+"\""+", effort: "+effort[6]+", start_time: 0, duration: "+duration[0]+"}'"
        
        self.c0="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"
        self.c1="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"
        self.c2="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"
        self.c3="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"
        self.c4="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"
        self.c5="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"
        self.c6="rosservice call gazebo/clear_joint_forces '{joint_name: "+"\""+robot_name+"::right_j0"+"\""+"}'"

    def _execute(self):
        self.make_services(self.R[-1],self.E[-1],self.D[-1])
        self.call_services()
        if(len(self.E)!=0):
            self.E.pop()
            self.D.pop()
            self.S.pop()
            self.R.pop()


    def call_services(self):
        os.system(self.s0+'&'+self.s1+'&'+self.s2+'&'+self.s3+'&'+self.s4+'&'+self.s5+'&'+self.s6)
        self.call_t=time.time()

    def _clear_jf():
        os.system(self.c0+'&'+self.c1+'&'+self.c2+'&'+self.c3+'&'+self.c4+'&'+self.c5+'&'+self.c6)


