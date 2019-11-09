import argparse
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION
import numpy as np
import rospy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

class arm(object):
    def __init__(self):
        # initialize our ROS node, registering it with the Master
        rospy.init_node('Hello_Sawyer', anonymous=True)
        # create an instance of intera_interface's Limb class
        self.limb = intera_interface.Limb('right')
        self.tip_name="right_gripper_tip"
        # get the right limb's current joint angles
        self.ini_angles = self.limb.joint_angles()
        rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state()
        try:
            self.gripper = intera_interface.Gripper('right')
        except ValueError:
            rospy.logerr("Could not detect a gripper attached to the robot.")
            return
        # print the current joint angles
        print("Initialized sawyer instance with angles",self.ini_angles)
    def grip(self,offset):
        current = self.gripper.get_position()
        self.gripper.set_position(current - offset)
    def reset(self):        
        # move to neutral pose
        self.limb.move_to_neutral()
        # get the right limb's current joint angles now that it is in neutral
        self.angles = self.limb.joint_angles()
        print("Reset joint angles ")
        # print the current joint angles again
        #print("Reset joint angles to ",self.angles)
    def orient_handcam(self):
        print("Changing position so that hand camera can see properly")
        self.angles['right_j5']=-1.59
        wave = {'right_j6': self.angles['right_j6'], 'right_j5': self.angles['right_j5'], 'right_j4': self.angles['right_j4'], 'right_j3': self.angles['right_j3'], 'right_j2': self.angles['right_j2'], 'right_j1': self.angles['right_j1'], 'right_j0': self.angles['right_j0']}
        self.limb.move_to_joint_positions(wave,timeout=3.0,threshold=0.005)
        rospy.sleep(0.2)
    def orient_headcam(self):
        print("Changing position so that head camera can see properly")
        self.angles['right_j5']=0.569
        wave = {'right_j6': self.angles['right_j6'], 'right_j5': self.angles['right_j5'], 'right_j4': self.angles['right_j4'], 'right_j3': self.angles['right_j3'], 'right_j2': self.angles['right_j2'], 'right_j1': self.angles['right_j1'], 'right_j0': self.angles['right_j0']}
        self.limb.move_to_joint_positions(wave,timeout=3.0,threshold=0.005)
        rospy.sleep(0.2)
    def robot_locked_away(self):
        ret=False
        if(self.angles['right_j0']<-0.732 or self.angles['right_j0']>0.55):
            return True
        '''
        if(self.angles['right_j1']<-1.8 or self.angles['right_j1']>0.002):
            return True
        if(self.angles['right_j2']<-1.2 or self.angles['right_j2']>1.4):
            return True
        if(self.angles['right_j2']<-0.06 or self.angles['right_j2']>2.8):
            return True
        '''
        return ret
    def joint_angles(self):
        j_a=[self.angles['right_j0'],self.angles['right_j1'],self.angles['right_j2'],self.angles['right_j3'],self.angles['right_j4'],self.angles['right_j5'],self.angles['right_j6']]
        j_a=np.reshape(j_a,[1,7])
        return j_a
    def move_arm(self,actions):
        self.angles['right_j0']+=actions[0]
        self.angles['right_j1']+=actions[1]
        self.angles['right_j2']+=actions[2]
        self.angles['right_j3']+=actions[3]
        self.angles['right_j4']+=actions[4]
        self.angles['right_j5']+=actions[5]
        self.angles['right_j6']+=actions[6]
        wave = {'right_j6': self.angles['right_j6'], 'right_j5': self.angles['right_j5'], 'right_j4': self.angles['right_j4'], 'right_j3': self.angles['right_j3'], 'right_j2': self.angles['right_j2'], 'right_j1': self.angles['right_j1'], 'right_j0': self.angles['right_j0']}
        self.limb.move_to_joint_positions(wave,timeout=2.0,threshold=0.005)
        rospy.sleep(0.2)
    def move_arm_to_pos(self,actions):
        self.angles['right_j0']=actions[0]
        self.angles['right_j1']=actions[1]
        self.angles['right_j2']=actions[2]
        self.angles['right_j3']=actions[3]
        self.angles['right_j4']=actions[4]
        self.angles['right_j5']=actions[5]
        self.angles['right_j6']=actions[6]
        wave = {'right_j6': self.angles['right_j6'], 'right_j5': self.angles['right_j5'], 'right_j4': self.angles['right_j4'], 'right_j3': self.angles['right_j3'], 'right_j2': self.angles['right_j2'], 'right_j1': self.angles['right_j1'], 'right_j0': self.angles['right_j0']}
        self.limb.move_to_joint_positions(wave,timeout=2.0,threshold=0.005)
        rospy.sleep(0.2)
    def move_arm_coord(self,tx,ty,tz): #Use inverse kinematics
        overhead_orientation = Quaternion(
                             x=-0.00142460053167,
                             y=0.999994209902,
                             z=-0.00177030764765,
                             w=0.00253311793936)
        block_poses = list()
        # The Pose of the block in its initial location.
        # You may wish to replace these poses with estimates
        # from a perception node.
        block_poses.append(Pose(position=Point(x=tx, y=ty, z=tz),orientation=overhead_orientation))
        joint_angles = self.limb.ik_request(block_poses[0], self.tip_name)
        print("joint angles calculated ",joint_angles)
        #self.limb.set_joint_position_speed(0.001)
        self.limb.move_to_joint_positions(joint_angles,timeout=2.0,threshold=0.005)
        self.limb.set_joint_position_speed(0.1)
    def get_xyz(self):
        c_end_pos = self.limb.endpoint_pose()
        c_val = c_end_pos.values()
        cex = c_val[0].x
        cey = c_val[0].y
        cez = c_val[0].z
        print("x ",cex,"y ",cey,"z ",cez)
        return cex,cey,cez
'''
def main():
    am=arm()
    cam=camera()
    am.reset()
    rospy.sleep(0.5)
    am.move_arm([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    #rospy.spin()# If not there still everything runs, but outputs a huge rospy interruption exception on ctrl+c event
    
if __name__ == '__main__':
    main()
'''