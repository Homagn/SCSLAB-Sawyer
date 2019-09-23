import numpy as np
import rospy
import pickle
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
import math
import commands
class sawyer(object):
    def __init__(self):
        self.am=arm()
        self.cam=camera()
        self.setup=False
        #self.cam_head=camera("head_camera")
        #t.sleep(5)
        #self.cam_hand=camera("right_hand_camera")
        self.cur_view="head"
        self.target_cartesian=self.am.limb.endpoint_pose()
        self.ref_joint_position1=self.am.limb.joint_angles()
        self.ref_joint_position2=self.am.limb.joint_angles()
        self.trials=0
        self.done=False
        self.reset()
    def position_block_randomly(self):
        #position the cube
        #posex=np.random.uniform(0.6,1.2)#0.45,0.75
        #posey=np.random.uniform(-0.20,0.35)#-0.15,0.20
        posex=np.random.uniform(0.35,0.80)#0.45,0.75
        posey=np.random.uniform(-0.35,0.35)#-0.15,0.20

        posex_case=np.random.uniform(0.8,1.1)#0.45,0.75
        posey_case=np.random.uniform(-0.30,0.20)#-0.15,0.20
        pose_orien=np.random.uniform(-0.50,1.50)#-0.15,0.20

        #off sets for the arm gripping position
        offset_x=-0.0008
        offset_y=0.0052
        offset_z=-0.823
        self.sim_targ_x=posex+offset_x
        self.sim_targ_y=posey+offset_y
        self.sim_targ_z=0.773+offset_z
        #set the cafe table position
        #cmd0="rosservice call /gazebo/set_model_state '{model_state: { model_name: cafe_table, pose: { position: { x: 0.965198, y: 0.004648, z: 0.773 }, orientation: {x: 0, y: 0, z: 0, w: 0.870604813099 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'"
        #os.system(cmd0)
        #randomly set wooden case position
        #change position_x, position_y and orientation_z
        #cmd1="rosservice call /gazebo/set_model_state '{model_state: { model_name: wooden_case_11, pose: { position: { x: "+str(posex_case)+", y: "+str(posey_case)+", z: 0.773 }, orientation: {x: 0, y: 0, z: "+str(pose_orien)+", w: 0.870604813099 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'"
        #os.system(cmd1)
        #randomly set wooden cube position
        cmd2="rosservice call /gazebo/set_model_state '{model_state: { model_name: wood_cube_10cm, pose: { position: { x: "+str(posex)+", y: "+str(posey)+", z: 0.873 }, orientation: {x: 0, y: 0.0, z: 0, w: 0.870604813099 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'"
        os.system(cmd2)
    def position_Real_block_randomly(self):
        #Assume that at this stage the robot is in a valid gripping position, about to grip an object
        self.am.gripper.open()
        self.am.limb.move_to_joint_positions(self.ref_joint_position1,timeout=2.0,threshold=0.005)
        self.am.limb.move_to_joint_positions(self.ref_joint_position2,timeout=2.0,threshold=0.005)
        self.am.gripper.close() #Grip the object with an assumed grip width of 3.5 cm
        self.am.gripper.close()
        rospy.sleep(0.5)
        self.am.limb.move_to_neutral()
        joint_file_trans=open('joint_record_trans.obj','r')
        joints_trans=pickle.load(joint_file_trans)

        joint_file=open('joint_record.obj','r')
        joints=pickle.load(joint_file)

        pos_file=open('end_pos.obj','r')
        positions=pickle.load(pos_file)


        sample_array=np.arange(len(joints))
        r_choice=random.choice(sample_array)
        one_joint=joints[r_choice]
        one_joint_trans=joints_trans[r_choice]
        one_pos=positions[r_choice]

        print("Got joint positions ",one_joint)
        self.am.limb.move_to_joint_positions(one_joint_trans,timeout=2.0,threshold=0.005)
        self.am.limb.move_to_joint_positions(one_joint,timeout=2.0,threshold=0.005)

        self.am.gripper.open()

        print("Current reloded end cube position ",one_pos)
        rospy.sleep(0.2)
        cartesian=one_pos#.values()
        #self.ref_joint_position=copy.copy(one_joint)
        self.ref_joint_position1=copy.copy(one_joint_trans)
        self.ref_joint_position2=copy.copy(one_joint)
        self.sim_targ_x=cartesian[0]#.x
        self.sim_targ_y=cartesian[1]#.y
        self.sim_targ_z=cartesian[2]#.z
        #am.grip(-3.0) #release the grip
        self.am.limb.move_to_joint_positions(one_joint_trans,timeout=2.0,threshold=0.005)
        self.am.limb.move_to_neutral()
        #self.am.grip(-3.0) #release the grip
    def reset(self,evaluate=False):
        self.am.reset()

        self.prev_sensor="head"
        self.head_sensor_adv=0.0
        self.hand_sensor_adv=0.0

        obs={}
        self.trials=0
        #if(random.random()>0.5): #Dont know why this was here
        self.am.orient_headcam()
        #self.position_block_randomly()
        if(evaluate==False):
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
            self.am.orient_headcam()
            obs["image"]=self.cam.see("head")
        if(self.cur_view=="hand"):
            self.am.orient_handcam()#Orient the hand camera to see the block.
            #self.cam.ts.registerCallback(self.cam.img_callback)
            obs["image"]=self.cam.see("hand")
        obs["joints"]=self.am.joint_angles()
        return obs
    def euc_dist(self,c,t):
        c_val=c.values()
        t_val=t.values()

        #status, output=commands.getstatusoutput("rosservice call /gazebo/get_link_state 'link_name: sawyer::right_l1'")
        #temporary hardcoding end effector position
        #x=0.596
        x=self.sim_targ_x
        #y=-0.065
        y=self.sim_targ_y
        #z=-0.050
        z=self.sim_targ_z
        #print("c quats",c_val[1])

        #roll=math.atan2(2*c_val[1].y*c_val[1].w+2*c_val[1].x*c_val[1].z, 1-2*c_val[1].y*c_val[1].y-2*c_val[1].z*c_val[1].z)
        #print("Got roll value as ",roll)
        #return math.sqrt((c_val[0].x-t_val[0].x)**2+(c_val[0].y-t_val[0].y)**2+(c_val[0].z-t_val[0].z)**2)
        return math.sqrt((c_val[0].x-x)**2+(c_val[0].y-y)**2+(c_val[0].z-z)**2)
    def step(self,actions,certainty):
        self.am.orient_headcam()
        self.prev_cartesian=copy.copy(self.am.limb.endpoint_pose())
        self.ini_cartesian=copy.copy(self.am.limb.endpoint_pose())
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
        
        reward_profile=np.zeros(len(actions))
        '''
        for i in range(len(actions[:-1])):
            action_profile=np.zeros(len(actions[:-1]))
            action_profile[i]= actions[:-1][i]
            action_profile_backtrack=-action_profile
            self.am.move_arm(action_profile)
            self.cur_cartesian=copy.copy(self.am.limb.endpoint_pose())
            reward=self.euc_dist(self.prev_cartesian,self.target_cartesian)-self.euc_dist(self.cur_cartesian,self.target_cartesian)
            self.am.move_arm(action_profile_backtrack)
            #self.prev_cartesian=copy.copy(self.am.limb.endpoint_pose())
            reward_profile[i]=reward
        '''

        self.am.move_arm(actions[:-1])
        self.cur_cartesian=copy.copy(self.am.limb.endpoint_pose())
        reward=self.euc_dist(self.prev_cartesian,self.target_cartesian)-self.euc_dist(self.cur_cartesian,self.target_cartesian)
        
        for i in range(len(actions[:-1])):
            reward_profile[i]=reward

        cur_distance=self.euc_dist(self.cur_cartesian,self.target_cartesian)
        gap_close=self.euc_dist(self.ini_cartesian,self.target_cartesian)-self.euc_dist(self.cur_cartesian,self.target_cartesian)
        


        obs={}
        if(self.cur_view=="head"):
            #reward_profile[-1]=self.head_sensor_adv
            reward_profile[-1]=certainty
        if(self.cur_view=="hand"):
            #reward_profile[-1]=self.hand_sensor_adv
            reward_profile[-1]=certainty
        if(actions[-1]==0):
            self.cur_view="head"
            #obpr=obs
            self.am.orient_headcam()#Orient the hand camera to see the block.
            obs["image"]=self.cam.see("head")
            obs["joints"]=self.am.joint_angles()

            self.head_sensor_adv+=0.5*gap_close
            #self.head_sensor_adv=gap_close
            if(self.prev_sensor=="hand"):
                self.head_sensor_adv=0.0
                #self.head_sensor_adv=self.hand_sensor_adv
            #reward_profile[-1]=self.head_sensor_adv
            self.prev_sensor="head"
            #print("Checking obs equality ",np.array_equal(obs,obpr))
        if(actions[-1]==1):
            self.cur_view="hand"
            #obpr=obs
            #self.cam.img_callback(self.cam.cur_image_hand,self.cam.cur_image_head)
            self.am.orient_handcam()#Orient the hand camera to see the block.
            obs["image"]=self.cam.see("hand")
            obs["joints"]=self.am.joint_angles()

            self.hand_sensor_adv+=0.5*gap_close
            #self.hand_sensor_adv=gap_close
            if(self.prev_sensor=="head"):
                self.hand_sensor_adv=0.0
                #self.hand_sensor_adv=self.head_sensor_adv
            #reward_profile[-1]=self.hand_sensor_adv
            self.prev_sensor="hand"
            #print("Checking obs equality ",np.array_equal(obs,obpr))
        #If distance closes, reward is positive
        self.trials+=1
        #Dont let the arm wander way too far
        print("Got reward profile",10*reward_profile)
        print("Changing tolerable distance based on switching")
        #self.start_cartesian=self.am.limb.endpoint_pose()
        #self.tolerable_dist=self.euc_dist(self.start_cartesian,self.target_cartesian)
        if(cur_distance>1.0*self.tolerable_dist):
            print("Wandered away too far ..")
            self.done=True
        #if(self.am.robot_locked_away()==True):
            #print("Robot in weird position ..")
            #self.done=True
    	if(cur_distance<0.2*self.tolerable_dist or self.trials==30):#prevent too much roaming around
            print("Goal reached !!")
            self.done=True
        return obs,10*reward_profile,self.done,cur_distance,self.tolerable_dist
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
