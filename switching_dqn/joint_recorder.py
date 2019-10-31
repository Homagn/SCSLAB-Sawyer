import argparse
import intera_interface
import numpy as np
import rospy
from arm_motion import arm
import csv
import random
import copy
import pickle
from collections import deque

def main():
    am=arm()
    
    joint_memory=deque(maxlen=2000)
    end_pos_memory=deque(maxlen=2000)
    ref_joint_position=am.limb.joint_angles()
    am.reset()
    #am.grip(-3.0)
    print("Welcome to recorder !, press s for save, r for replay and just enter to stop ")
    while True:
        com = raw_input('Command:').split()
        if len(com) == 0:
            break
        elif com[0] == 'b':
            am.reset()
            am.grip(-3.0)
        elif com[0] == 's':
            end_pose=copy.copy(am.limb.endpoint_pose())
            ep_vals=end_pose.values()

            angles = am.limb.joint_angles()

            joint_memory.append(angles)
            end_pos_memory.append([ep_vals[0].x,ep_vals[0].y,ep_vals[0].z])

            fname = 'joint_record.obj'
            file1 = open(fname, 'w')
            pickle.dump(joint_memory,file1)
            file1.close()
            #writer = csv.writer(file1)
            #fields1=[angles]
            #writer.writerow(fields1)
            #file1.close()

            fname = 'end_pos.obj'
            file1 = open(fname, 'w')
            pickle.dump(end_pos_memory,file1)
            file1.close()

        elif com[0] == 'r':
            options=[]
            am.limb.move_to_joint_positions(ref_joint_position,timeout=2.0,threshold=0.005)
            #Assume that at this stage the robot is in a valid gripping position, about to grip an object
            am.grip(0.03) #Grip the object with an assumed grip width of 3.5 cm
            am.limb.move_to_neutral()
            joint_file=open('joint_record.obj','r')
            joints=pickle.load(joint_file)

            pos_file=open('end_pos.obj','r')
            positions=pickle.load(pos_file)


            sample_array=np.arange(len(joints))
            r_choice=random.choice(sample_array)
            one_joint=joints[r_choice]
            one_pos=positions[r_choice]

            print("Got joint positions ",one_joint)
            am.limb.move_to_joint_positions(one_joint,timeout=2.0,threshold=0.005)
            ref_joint_position=copy.copy(one_joint)
            print("Current reloded end cube position ",one_pos)
            rospy.sleep(0.2)
            am.limb.move_to_neutral()
            am.grip(-3.0) #release the grip
if __name__ == '__main__':
    main()
    
