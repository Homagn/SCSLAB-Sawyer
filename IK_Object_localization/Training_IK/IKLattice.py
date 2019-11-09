# This is the training class for learning inverse kinematics
# Perform inverse kinematics by dense matching

import numpy as np
import rospy
from sawyer import sawyer
import random
from random import sample
from collections import deque
import pickle


import copy
import math

from sklearn.neighbors import KNeighborsRegressor


def get_xyz(env):
    c_end_pos = env.am.limb.endpoint_pose()
    c_val = c_end_pos.values()
    cex = c_val[0].x
    cey = c_val[0].y
    cez = c_val[0].z
    print("x ",cex,"y ",cey,"z ",cez)
    return cex,cey,cez
def write_file(fname,data):
    file1 = open(fname, 'w')
    pickle.dump(data,file1)
    file1.close()
def collect_data():
    env=sawyer()
    x = []
    y = []
    z = []
    lattice = []
    try:
        fx = open('xlist.obj','r')
        x = pickle.load(fx)

        fy = open('ylist.obj','r')
        y = pickle.load(fy)

        fz = open('zlist.obj','r')
        z = pickle.load(fz)

        fl = open('lattice.obj','r')
        lattice = pickle.load(fl)

    except:
        print("lattice files were not found")
    bj0 = -1.5
    bj1 = -0.5
    bj3 = -1.5
    bj5 = -2.0
    for i in range(2,15):
        for j in range(15):
            for k in range(20):
                for l in range(30):
                    x1,y1,z1 = get_xyz(env)
                    x.append(x1)
                    y.append(y1)
                    z.append(z1)
                    x.sort()
                    y.sort()
                    z.sort()

                    a = x.index(x1)
                    b = y.index(y1)
                    c = z.index(z1)

                    cur_joints = env.am.joint_angles()
                    #Change the mapping in the lattice after new point insertion
                    for e in lattice:
                        if a in e["coord_idx"]:
                            e["coord_idx"] = [a+1,b,c]
                        if b in e["coord_idx"]:
                            e["coord_idx"] = [a,b+1,c]
                        if c in e["coord_idx"]:
                            e["coord_idx"] = [a,b,c+1]

                    lattice_point = {}
                    lattice_point["coord_idx"] = [a,b,c]
                    lattice_point["angles"] = cur_joints
                    lattice.append(lattice_point)

                    joint_pos = [bj0+i*0.1, bj1+j*0.1,0.0,bj3+k*0.1,0.0,bj5+l*0.1,0.0]
                    env.am.move_arm_to_pos(joint_pos)
                    print("Number of lattice points so far ",len(lattice))
                    print("number of x_coords ",len(x))
                    print("number of y_coords ",len(y))
                    print("number of z_coords ",len(z))
                    print("i ",i,"j ",j,"k ",k,"l ",l)

                    write_file('xlist.obj',x)
                    write_file('ylist.obj',y)
                    write_file('zlist.obj',z)

                    write_file('lattice.obj',lattice)

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
#might try map reduce to find the closest element also
def inference():
    try:
        fx = open('xlist.obj','r')
        x = pickle.load(fx)

        fy = open('ylist.obj','r')
        y = pickle.load(fy)

        fz = open('zlist.obj','r')
        z = pickle.load(fz)

        fl = open('lattice.obj','r')
        lattice = pickle.load(fl)

    except:
        print("lattice files were not found")
        return
    env=sawyer()
    #x1,y1,z1= get_xyz(env)
    deltas = []
    for i in range(7):
        if(i==0 or i==1 or i==3 or i==5):
            deltas.append(0.5*random.uniform(-1,1))
        else:
            deltas.append(0.0)
    print("original deltas ",deltas)
    env.am.move_arm(deltas)
    x2,y2,z2 = get_xyz(env)
    env.am.reset()

    print("lattice matching ")

    target_joints = []
    minimum = 100
    # Doing a BFS in the whole lattice
    for l in lattice:
        [c1,c2,c3] = l["coord_idx"]
        [x1,y1,z1] = [x[c1],y[c2],z[c3]]
        dist = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        if(dist<=minimum):
            target_joints = l["angles"]
            minimum = dist
    print("Closest solution found at tolerance ",math.sqrt(dist))
    env.am.move_arm_to_pos(target_joints)

if __name__ == '__main__':
    collect_data()





