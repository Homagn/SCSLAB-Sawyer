import numpy as np
import rospy
from arm_motion import arm
from wait_for_camera import camera
import math
import time as t
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import random 
import copy
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
class sawyer(object):
    def __init__(self):
        self.am=arm()
        self.cam=camera()
        self.setup=True
        #self.cam_head=camera("head_camera")
        #t.sleep(5)
        #self.cam_hand=camera("right_hand_camera")
        self.cur_view="head"
        self.target_cartesian=self.am.limb.endpoint_pose()
        self.ref_joint_position=self.am.limb.joint_angles()
        self.trials=0
        self.done=False
        self.reset()
    def position_block_randomly(self,x_grid=-1,y_grid=-1):
        #position the cube
        if x_grid==-1 and y_grid==-1:
            posex=np.random.uniform(0.54,0.90)
            posey=np.random.uniform(-0.18,0.39)
        else:
            posex = 0.54+0.01*x_grid
            posey = -0.18+0.01*y_grid
        #off sets for the arm gripping position
        offset_x=-0.042
        offset_y=0.004
        offset_z=-0.823
        self.sim_targ_x=posex+offset_x
        self.sim_targ_y=posey+offset_y
        self.sim_targ_z=0.773+offset_z
        '''
        cmd2="rosservice call /gazebo/set_model_state '{model_state: { model_name: wood_cube_10cm, pose: { position: { x: "+str(posex)+", y: "+str(posey)+", z: 0.773 }, orientation: {x: 0, y: 0.0, z: 0, w: 0.870604813099 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'"
        os.system(cmd2)
        '''
        self.teleport("wood_cube_10cm",posex,posey)
        return posex, posey
    def teleport(self, model_name, x, y): #move gazebo objects extremely fast (like a discreet step)
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.773
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = 0.0
        state_msg.pose.orientation.w = 0.870604813099
        #rospy.wait_for_service('/gazebo/set_model_state') #already called in init
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg)

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
    def position_table(self):
        posex = 0.845830
        posey = 0.069945

        cmd2="rosservice call /gazebo/set_model_state '{model_state: { model_name: cafe_table, pose: { position: { x: "+str(posex)+", y: "+str(posey)+", z: 0.0 }, orientation: {x: 0, y: 0.0, z: 0, w: 0.0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'"
        os.system(cmd2)
    def position_Real_block_randomly(self):
        #Assume that at this stage the robot is in a valid gripping position, about to grip an object
        self.am.limb.move_to_joint_positions(self.ref_joint_position,timeout=2.0,threshold=0.005)
        self.am.grip(0.029) #Grip the object with an assumed grip width of 3.5 cm
        joint_file=open('saved_positions/joint_record.obj','r')
        joints=pickle.load(joint_file)

        pos_file=open('saved_positions/end_pos.obj','r')
        positions=pickle.load(pos_file)


        sample_array=np.arange(len(joints))
        r_choice=random.choice(sample_array)
        one_joint=joints[r_choice]
        one_pos=positions[r_choice]

        print("Got joint positions ",one_joint)
        self.am.limb.move_to_joint_positions(one_joint,timeout=2.0,threshold=0.005)
        print("Current reloded end cube position ",one_pos)
        rospy.sleep(0.2)
        cartesian=one_pos.values()
        self.ref_joint_position=copy.copy(one_joint)
        self.sim_targ_x=cartesian[0].x
        self.sim_targ_y=cartesian[0].y
        self.sim_targ_z=cartesian[0].z
        #am.grip(-3.0) #release the grip
        self.am.limb.move_to_neutral()
        self.am.grip(-3.0) #release the grip
    def reset(self,table_view = False):
        self.am.reset()
        if table_view:
            self.am.move_arm([-0.1,0.0,0.0,-1.0,0.0,-0.6,0.0])
        obs={}
        self.trials=0
        #if(random.random()>0.5): #Dont know why this was here
        if not table_view:
            self.am.orient_headcam()
        self.position_table()
        self.position_block_randomly()
        
        #self.position_Real_block_randomly()
        print("Done block position change ")
        self.done=False
        self.start_cartesian=self.am.limb.endpoint_pose()
        self.tolerable_dist=self.euc_dist(self.start_cartesian,self.target_cartesian)
        #print("start cartesian ",self.start_cartesian)
        #print("target cartesian ",self.target_cartesian)
        print("##################")
        print("Tolerable distance set as ",self.tolerable_dist)
        print("##################")
        if(self.setup==False):
            #position the table
            try:
                cmd1="rosservice call /gazebo/set_model_state '{model_state: { model_name: cafe_table, pose: { position: { x: 0.72, y: -0.02 ,z: 0.0 }, orientation: {x: 0, y: 0.0, z: 0, w: 0.870604813099 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'"
                os.system(cmd1)
            except:
                print("Either this is not a simulation or model names are different")
            self.setup=True
        if(self.cur_view=="head"):
            #self.cam.ts.registerCallback(self.cam.img_callback)
            if not table_view:
                self.am.orient_headcam()
            obs["image"]=self.cam.see("head",show = False)
        if(self.cur_view=="hand"):
            self.am.orient_handcam()#Orient the hand camera to see the block.
            #self.cam.ts.registerCallback(self.cam.img_callback)
            obs["image"]=self.cam.see("hand",show = False)
        obs["joints"]=self.am.joint_angles()
        return obs
    def euc_dist(self,c,t):
        c_val=c.values()
        t_val=t.values()
        #temporary hardcoding end effector position
        #x=0.596
        x=self.sim_targ_x
        #y=-0.065
        y=self.sim_targ_y
        #z=-0.050
        z=self.sim_targ_z
        #return math.sqrt((c_val[0].x-t_val[0].x)**2+(c_val[0].y-t_val[0].y)**2+(c_val[0].z-t_val[0].z)**2)
        return math.sqrt((c_val[0].x-x)**2+(c_val[0].y-y)**2+(c_val[0].z-z)**2)
    def step(self,actions):
        self.am.orient_headcam()
        self.prev_cartesian=copy.copy(self.am.limb.endpoint_pose())
        #print("My initial distance ",self.euc_dist(self.prev_cartesian,self.target_cartesian))
        #print("check simulation for arm change ")
        #t.sleep(2)
        if(np.count_nonzero(actions[:-1])==0):
            print("All zero actions, this is bad so taking a random action")
            a=np.random.randint(3,size=7)
            act_rand=[]
            for i in a: #Last one for the switch
                if(i%3==0):
                    act_rand.append(0.0)
                if(i%3==1):
                    act_rand.append(0.05)
                if(i%3==2):
                    act_rand.append(-0.05)
            #self.am.move_arm(act_rand) #optional
            self.done=True
        self.am.move_arm(actions[:-1])
        #t.sleep(2)
        self.cur_cartesian=copy.copy(self.am.limb.endpoint_pose())
        #print("My final distance ",self.euc_dist(self.cur_cartesian,self.target_cartesian))
        #t.sleep(2)
        reward=self.euc_dist(self.prev_cartesian,self.target_cartesian)-self.euc_dist(self.cur_cartesian,self.target_cartesian)
        cur_distance=self.euc_dist(self.cur_cartesian,self.target_cartesian)
        obs={}
        if(actions[-1]==0):
            self.cur_view="head"
            #obpr=obs
            self.am.orient_headcam()#Orient the hand camera to see the block.
            obs["image"]=self.cam.see("head")
            obs["joints"]=self.am.joint_angles()
            #print("Checking obs equality ",np.array_equal(obs,obpr))
        if(actions[-1]==1):
            self.cur_view="hand"
            #obpr=obs
            #self.cam.img_callback(self.cam.cur_image_hand,self.cam.cur_image_head)
            self.am.orient_handcam()#Orient the hand camera to see the block.
            obs["image"]=self.cam.see("hand")
            obs["joints"]=self.am.joint_angles()
            #print("Checking obs equality ",np.array_equal(obs,obpr))
        #If distance closes, reward is positive
        self.trials+=1
        #Dont let the arm wander way too far
        print("Got reward ",10*reward)
        print("Changing tolerable distance based on switching")
        #self.start_cartesian=self.am.limb.endpoint_pose()
        #self.tolerable_dist=self.euc_dist(self.start_cartesian,self.target_cartesian)
        if(cur_distance>1.2*self.tolerable_dist):
            print("Wandered away too far ..")
            self.done=True
        #if(self.am.robot_locked_away()==True):
            #print("Robot in weird position ..")
            #self.done=True
    	if(cur_distance<0.5*self.tolerable_dist or self.trials==49):#prevent too much roaming around
            print("Goal reached !!")
            self.done=True
        return obs,10*reward,self.done,cur_distance,self.tolerable_dist
'''
def main():
    sw=sawyer()
    
    rospy.sleep(0.2)
    ob,reward,done=sw.step([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    print("shape of image ",ob.shape)
    print("Got reward ",reward)
    print("Done? ",done)

if __name__ == '__main__':
    main()
'''
