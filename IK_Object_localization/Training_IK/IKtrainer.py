# This is the training class for learning inverse kinematics
# Use neural network to learn IK from sampled datapoints
import numpy as np
import rospy
from sawyer import sawyer
import random
from random import sample
from collections import deque
import pickle

from keras.models import Sequential
from keras.layers import Concatenate,Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.models import model_from_json
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

import copy

from sklearn.neighbors import KNeighborsRegressor


def get_xyz(env):
    c_end_pos = env.am.limb.endpoint_pose()
    c_val = c_end_pos.values()
    cex = c_val[0].x
    cey = c_val[0].y
    cez = c_val[0].z
    print("x ",cex,"y ",cey,"z ",cez)
    return cex,cey,cez

def collect_data(num_points):
    env=sawyer()
    delta_memory=deque(maxlen=num_points)
    xyz_memory=deque(maxlen=num_points)
    try:
        d_file = open('deltas.obj','r')
        delta_memory = pickle.load(d_file)

        p_file = open('xyzpos.obj','r')
        xyz_memory = pickle.load(p_file)
    except:
        print("Files were not found")

    for n in range(num_points):
        if(n%19==0):
            env.am.reset()
            num_samples = random.randint(1,4)
            joint_list = sample([0,1,3,5],num_samples)
            print("sampled joints ",joint_list)
        x1,y1,z1= get_xyz(env)
        deltas = []
        for i in range(7):
            if(i in joint_list):
                deltas.append(0.15*random.uniform(-1,1))
            else:
                deltas.append(0.0)
        
        cur_joint_angles = env.am.joint_angles().tolist() #get joint angles before moving the arm
        env.am.move_arm(deltas)
        x2,y2,z2 = get_xyz(env)
        instance = [x1,y1,z1,x2,y2,z2]
        instance.extend(cur_joint_angles[0])

        reverse_instance = [x2,y2,z2,x1,y1,z1]
        cur_joint_angles = env.am.joint_angles().tolist()
        reverse_instance.extend(cur_joint_angles[0])


        xyz_memory.append(instance)
        xyz_memory.append(reverse_instance)
        print("last xyz memory ",xyz_memory[-1])
        delta_memory.append(deltas)
        delta_memory.append([-item for item in deltas])
        print("last delta ",delta_memory[-1])

        fname = 'deltas.obj'
        file1 = open(fname, 'w')
        pickle.dump(delta_memory,file1)
        file1.close()

        fname = 'xyzpos.obj'
        file2 = open(fname, 'w')
        pickle.dump(xyz_memory,file2)
        file2.close()
        print("length of memory ",len(xyz_memory))

def build_model():
    inp= Input(shape=(13,))
    x = Dense(100, activation='tanh')(inp)
    x = Dense(100, activation='tanh')(x)
    x = Dense(100, activation='tanh')(x)
    x = Dense(50, activation='tanh')(x)
    x = Dense(25, activation='tanh')(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def build_special_model():
    inp1 = Input(shape=(13,))
    a = Dense(100, activation='tanh')(inp1)
    a = Dense(100, activation='tanh')(a)
    a = Dense(100, activation='tanh')(a)
    a = Dense(50, activation='tanh')(a)
    a = Dense(25, activation='tanh')(a)
    out1 = Dense(1, activation='linear')(a)

    inp2 = concatenate([inp1,out1])
    b = Dense(100, activation='tanh')(inp2)
    b = Dense(100, activation='tanh')(b)
    b = Dense(100, activation='tanh')(b)
    b = Dense(50, activation='tanh')(b)
    b = Dense(25, activation='tanh')(b)
    out2 = Dense(1, activation='linear')(b)

    inp3 = concatenate([inp1,out1,out2])
    c = Dense(100, activation='tanh')(inp3)
    c = Dense(100, activation='tanh')(c)
    c = Dense(100, activation='tanh')(c)
    c = Dense(50, activation='tanh')(c)
    c = Dense(25, activation='tanh')(c)
    out3 = Dense(1, activation='linear')(c)

    inp4 = concatenate([inp1,out1,out2,out3])
    d = Dense(100, activation='tanh')(inp4)
    d = Dense(100, activation='tanh')(d)
    d = Dense(100, activation='tanh')(d)
    d = Dense(50, activation='tanh')(d)
    d = Dense(25, activation='tanh')(d)
    out4 = Dense(1, activation='linear')(d)

    final = concatenate([out1,out2,out3,out4])
    e = Dense(100, activation='tanh')(final)
    e = Dense(100, activation='tanh')(e)
    e = Dense(100, activation='tanh')(e)
    e = Dense(50, activation='tanh')(e)
    e = Dense(25, activation='tanh')(e)
    output = Dense(4, activation='linear')(e)

    model = Model(inputs=inp1, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def load_data(delta_file, xyzpos_file):
    d_file = open(delta_file,'r')
    delta = pickle.load(d_file)
    print("number of training points ",len(delta))

    p_file = open(xyzpos_file,'r')
    xyzpos = pickle.load(p_file)

    x_train = np.array(xyzpos)
    print(np.shape(x_train))
    x_train = np.reshape(x_train, (len(xyzpos),13))
    x_params = []
    y_params = []
    for i in range(13):
        feature = x_train[:,i]
        #print("got original feature ",feature)
        #print("max of feature ",np.max(feature))
        #print("min of feature ",np.min(feature))
        #print("mean of feature ",np.mean(feature))
        x_params.append([np.max(feature),np.min(feature),np.mean(feature)])

        feature = (x_train[:,i]-np.mean(x_train[:,i]))/(np.max(x_train[:,i])-np.min(x_train[:,i]))
        x_train[:,i] = feature

        #print("normalized feature ",feature)

    y_train = np.zeros((len(delta),4))
    for i in range(len(delta)):
        #print("delta[i] ",delta[i])
        y_train[i][0] = delta[i][0]
        y_train[i][1] = delta[i][1]
        y_train[i][2] = delta[i][3]
        y_train[i][3] = delta[i][5]

    for i in range(4):
        feature = y_train[:,i]
        #print("got original feature ",feature)
        #print("max of feature ",np.max(feature))
        #print("min of feature ",np.min(feature))
        #print("mean of feature ",np.mean(feature))
        y_params.append([np.max(feature),np.min(feature),np.mean(feature)])
        
        feature = (y_train[:,i]-np.mean(y_train[:,i]))/(np.max(y_train[:,i])-np.min(y_train[:,i]))
        y_train[:,i] = feature


        #print("normalized feature ",feature)

    return x_train, y_train, x_params, y_params

def train_models():
    x_train, y_train,_,_ = load_data('deltas.obj','xyzpos.obj')
    model = build_special_model()
    model.fit(x_train,y_train,batch_size = 16, epochs=150, verbose =1)
    model.save_weights("ikweights_special.h5")

    '''
    model = build_model()
    model.fit(x_train,y_train[:,0],batch_size = 16, epochs=150, verbose =1)
    model.save_weights("ikweights1.h5")
    
    model = build_model()
    model.fit(x_train,y_train[:,1],batch_size = 16, epochs=150, verbose =1)
    model.save_weights("ikweights2.h5")

    model = build_model()
    model.fit(x_train,y_train[:,2],batch_size = 16, epochs=150, verbose =1)
    model.save_weights("ikweights3.h5")

    model = build_model()
    model.fit(x_train,y_train[:,3],batch_size = 16, epochs=150, verbose =1)
    model.save_weights("ikweights4.h5")
    '''

def test_model():
    #Testing NN model
    _,_,x_param,y_param = load_data('deltas.obj','xyzpos.obj')

    model1 = build_model()
    model1.load_weights("ikweights1.h5")

    model2 = build_model()
    model2.load_weights("ikweights2.h5")

    model3 = build_model()
    model3.load_weights("ikweights3.h5")

    model4 = build_model()
    model4.load_weights("ikweights4.h5")

    env=sawyer()
    x1,y1,z1= get_xyz(env)
    deltas = []
    for i in range(7):
        if(i==0 or i==1 or i==3 or i==5):
            deltas.append(0.2*random.uniform(-1,1))
        else:
            deltas.append(0.0)
    print("original deltas ",deltas)
    #print("original scaled deltas ",(np.array(deltas) - y_param[2])/(y_param[0]-y_param[1]))

    env.am.move_arm(deltas)
    x2,y2,z2 = get_xyz(env)
    print("moving back to original position and testing ikmodel")
    env.am.reset()

    instance = [x1,y1,z1,x2,y2,z2]
    cur_joint_angles = env.am.joint_angles().tolist()
    instance.extend(cur_joint_angles[0])

    for i in range(len(instance)):
        instance[i] = (instance[i]- x_param[i][2])/(x_param[i][0]-x_param[i][1]) 
    
    instance = np.array(instance)

    output1 = model1.predict(instance.reshape(1,13))[0]
    output2 = model2.predict(instance.reshape(1,13))[0]
    output3 = model3.predict(instance.reshape(1,13))[0]
    output4 = model4.predict(instance.reshape(1,13))[0]

    output = [output1,output2,output3,output4]

    for i in range(len(output)):
        output[i] = output[i] *(y_param[i][0]-y_param[i][1]) + y_param[i][2]

    pred_delta = np.array(output)

    p_delta = [0 for i in range(7)]
    p_delta[0] = pred_delta[0]
    p_delta[1] = pred_delta[1]
    p_delta[3] = pred_delta[2]
    p_delta[5] = pred_delta[3]
    print("pred_delta ",pred_delta)
    env.am.move_arm(p_delta)
    
def test_special_model():
    #Testing NN model
    _,_,x_param,y_param = load_data('deltas.obj','xyzpos.obj')

    special_model = build_special_model()
    special_model.load_weights("ikweights_special.h5")

    env=sawyer()
    x1,y1,z1= get_xyz(env)
    deltas = []
    for i in range(7):
        if(i==0 or i==1 or i==3 or i==5):
            deltas.append(0.2*random.uniform(-1,1))
        else:
            deltas.append(0.0)
    print("original deltas ",deltas)
    #print("original scaled deltas ",(np.array(deltas) - y_param[2])/(y_param[0]-y_param[1]))

    env.am.move_arm(deltas)
    x2,y2,z2 = get_xyz(env)
    print("moving back to original position and testing ikmodel")
    env.am.reset()

    instance = [x1,y1,z1,x2,y2,z2]
    cur_joint_angles = env.am.joint_angles().tolist()
    instance.extend(cur_joint_angles[0])

    for i in range(len(instance)):
        instance[i] = (instance[i]- x_param[i][2])/(x_param[i][0]-x_param[i][1]) 
    
    instance = np.array(instance)

    output = special_model.predict(instance.reshape(1,13))[0]

    for i in range(len(output)):
        output[i] = output[i] *(y_param[i][0]-y_param[i][1]) + y_param[i][2]

    pred_delta = np.array(output)

    p_delta = [0 for i in range(7)]
    p_delta[0] = pred_delta[0]
    p_delta[1] = pred_delta[1]
    p_delta[3] = pred_delta[2]
    p_delta[5] = pred_delta[3]
    print("pred_delta ",pred_delta)
    env.am.move_arm(p_delta)

def test_model_interpolation():
    #Testing NN model
    _,_,x_param,y_param = load_data('deltas.obj','xyzpos.obj')

    special_model = build_special_model()
    special_model.load_weights("ikweights_special.h5")

    env=sawyer()
    x1,y1,z1= get_xyz(env)
    deltas = []
    for i in range(7):
        if(i==0 or i==1 or i==3 or i==5):
            deltas.append(random.uniform(-1,1))
        else:
            deltas.append(0.0)
    print("original deltas ",deltas)
    #print("original scaled deltas ",(np.array(deltas) - y_param[2])/(y_param[0]-y_param[1]))

    env.am.move_arm(deltas)
    x2,y2,z2 = get_xyz(env)
    print("moving back to original position and testing ikmodel")
    env.am.reset()

    res = 0.2
    xi = x1
    yi = y1
    zi = z1
    
    for k in range(1,int(1/res),1):

        xt = res*(x2-x1)*k
        yt = res*(y2-y1)*k
        zt = res*(z2-z1)*k

        instance = [xi,yi,zi,xt,yt,zt]
        cur_joint_angles = env.am.joint_angles().tolist()
        instance.extend(cur_joint_angles[0])

        for i in range(len(instance)):
            instance[i] = (instance[i]- x_param[i][2])/(x_param[i][0]-x_param[i][1]) 
        
        instance = np.array(instance)

        output = special_model.predict(instance.reshape(1,13))[0]

        for i in range(len(output)):
            output[i] = output[i] *(y_param[i][0]-y_param[i][1]) + y_param[i][2]

        pred_delta = np.array(output)

        p_delta = [0 for i in range(7)]
        p_delta[0] = pred_delta[0]
        p_delta[1] = pred_delta[1]
        p_delta[3] = pred_delta[2]
        p_delta[5] = pred_delta[3]
        print("pred_delta ",pred_delta)
        xi = copy.copy(xt)
        yi = copy.copy(yt)
        zi = copy.copy(zt)
        env.am.move_arm(p_delta)

    
if __name__ == '__main__':
    collect_data(50000)

    
    #train_models()

    #test_special_model()

    #test_model_interpolation()
    

    #test_model()
    

    '''
    #Trying kNN regressor 
    x_train, y_train = load_data('deltas.obj','xyzpos.obj')
    neigh = KNeighborsRegressor(n_neighbors=500)
    neigh.fit(x_train, y_train)

    env=sawyer()
    x1,y1,z1= get_xyz(env)
    deltas = []
    for i in range(7):
        if(i==0 or i==1 or i==3 or i==5):
            deltas.append(0.2*random.uniform(-1,1))
        else:
            deltas.append(0.0)
    print("original deltas ",deltas)
    env.am.move_arm(deltas)
    x2,y2,z2 = get_xyz(env)
    print("moving back to original position and testing ikmodel")
    env.am.reset()
    pred_delta = neigh.predict(np.array([x1,y1,z1,x2,y2,z2]).reshape(1,6))[0]
    print("pred_delta ",pred_delta)
    p_delta = [0 for i in range(7)]
    p_delta[0] = pred_delta[0]
    p_delta[1] = pred_delta[1]
    p_delta[3] = pred_delta[2]
    p_delta[5] = pred_delta[3]
    env.am.move_arm(p_delta)
    '''




