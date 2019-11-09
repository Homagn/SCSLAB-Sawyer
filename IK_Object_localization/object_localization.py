from sawyer import sawyer
import time as t
import pickle
import numpy as np

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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def write_file(fname,data):
    file1 = open(fname, 'w')
    pickle.dump(data,file1)
    file1.close()
def build_model():
    inp= Input(shape=(4,))
    x = Dense(100, activation='tanh')(inp)
    x = Dense(100, activation='tanh')(x)
    x = Dense(100, activation='tanh')(x)
    x = Dense(50, activation='tanh')(x)
    x = Dense(25, activation='tanh')(x)
    out = Dense(2, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def load_data(xy_file, imagexy_file):
    fxy = open(xy_file,'r')
    xy = pickle.load(fxy)
    print("number of training points ",len(xy))

    fixy = open(imagexy_file,'r')
    imagexy = pickle.load(fixy)

    x_train = np.array(imagexy)
    x_train = np.reshape(x_train, (len(imagexy),4))
    print("x_train ",np.shape(x_train))
    print("x_train sample ",x_train[1])

    y_train = np.array(xy)
    y_train = np.reshape(y_train, (len(xy),2))
    print("y_train ",np.shape(y_train))
    print("y_train sample ",y_train[1])

    x_params = []
    y_params = []
    # Feature normalization if you want
    '''
    for i in range(4):
        feature = x_train[:,i]
        #print("got original feature ",feature)
        #print("max of feature ",np.max(feature))
        #print("min of feature ",np.min(feature))
        #print("mean of feature ",np.mean(feature))
        x_params.append([np.max(feature),np.min(feature),np.mean(feature)])
        feature = (x_train[:,i]-np.mean(x_train[:,i]))/(np.max(x_train[:,i])-np.min(x_train[:,i]))
        x_train[:,i] = feature
        #print("normalized feature ",feature)
    '''
    '''
    for i in range(2):
        feature = y_train[:,i]
        #print("got original feature ",feature)
        #print("max of feature ",np.max(feature))
        #print("min of feature ",np.min(feature))
        #print("mean of feature ",np.mean(feature))
        y_params.append([np.max(feature),np.min(feature),np.mean(feature)])
        feature = (y_train[:,i]-np.mean(y_train[:,i]))/(np.max(y_train[:,i])-np.min(y_train[:,i]))
        y_train[:,i] = feature
        #print("normalized feature ",feature)
    '''
    return x_train, y_train, x_params, y_params
def train_model(num_epochs):
    x_train, y_train,_,_ = load_data('xy.obj','imagexy.obj')
    model = build_model()
    model.fit(x_train,y_train,batch_size = 16, epochs=num_epochs, verbose =1)
    model.save_weights("camera_callibration.h5")
def test_shallow_model():
    # Dont normalize when using shallow techniques from scikit
    x_train, y_train,x_param,y_param = load_data('xy.obj','imagexy.obj')
    #neigh = KNeighborsRegressor(n_neighbors=10)
    #neigh.fit(x_train, y_train)
    regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    print("Trained shallow regressor")

    regr.fit(x_train, y_train)

    env=sawyer()
    env.reset(table_view = True)
    x,y = env.position_block_randomly()
    t.sleep(3)
    print("random block position ",x," ",y)

    view = "head"
    image = env.cam.see(view,show = False, size = [1280,800])
    ix1,iy1 = env.cam.detect_box(image,view,show_detection=False)

    view  = "hand"
    image = env.cam.see(view,show = False, size = [1280,800])
    ix2,iy2 = env.cam.detect_box(image,view,show_detection=False)

    instance = [ix1,ix2,iy1,iy2]

    instance = np.array(instance)
    output = regr.predict(instance.reshape(1,4))[0]

    pred_x = output[0]
    pred_y = output[1]
    z = 0.773-0.823
    print("Model estimated objects actual location to be ",pred_x," ",pred_y)
    env.am.move_arm_coord(pred_x,pred_y,z)

def test_model():
    #Testing NN model
    _,_,x_param,y_param = load_data('xy.obj','imagexy.obj')

    model = build_model()
    model.load_weights("camera_callibration.h5")

    env=sawyer()
    env.reset(table_view = True)
    x,y = env.position_block_randomly()
    t.sleep(3)
    print("random block position ",x," ",y)

    view = "head"
    image = env.cam.see(view,show = False, size = [1280,800])
    ix1,iy1 = env.cam.detect_box(image,view,show_detection=False)

    view  = "hand"
    image = env.cam.see(view,show = False, size = [1280,800])
    ix2,iy2 = env.cam.detect_box(image,view,show_detection=False)

    instance = [ix1,ix2,iy1,iy2]
    for i in range(len(instance)):
        instance[i] = (instance[i]- x_param[i][2])/(x_param[i][0]-x_param[i][1]) 
    instance = np.array(instance)
    output = model.predict(instance.reshape(1,4))[0]

    pred_x = output[0]
    pred_y = output[1]
    z = 0.773-0.823
    print("Model estimated objects actual location to be ",pred_x," ",pred_y)
    env.am.move_arm_coord(pred_x,pred_y,z)

def collect_data(numpoints):
    env = sawyer()
    xy = []
    imagexy = []
    try:
        fxy = open('xy.obj','r')
        xy = pickle.load(fxy)

        fixy = open('imagexy.obj','r')
        imagexy = pickle.load(fixy)
    except:
        print("Files were not found ")

    env.reset(table_view = True)
    for i in range(36):
        for j in range(57):
            x,y = env.position_block_randomly(x_grid = i, y_grid = j)
            print("random block position ",x," ",y)

            view = "head"
            image = env.cam.see(view,show = False, size = [1280,800])
            ix1,iy1 = env.cam.detect_box(image,view)

            view  = "hand"
            image = env.cam.see(view,show = False, size = [1280,800])
            ix2,iy2 = env.cam.detect_box(image,view)

            xy.append([x,y])
            imagexy.append([ix1,ix2,iy1,iy2])
            print("Length of data collected ",len(xy))
            write_file('xy.obj',xy)
            write_file('imagexy.obj',imagexy)

        
    #t.sleep(20)
if __name__ == '__main__':
    #collect_data(50000) #For collecting camera calibration data (find a mapping between image x,y and real x,y)
    #train_model(10) #Train a NN model based on collected data
    #test_model() #Test the trained NN model
    test_shallow_model() #Fit a shallow model from scikit learn on the collected data for 
    # learning mapping from image x,y to world ooordinate x,y and use the shallow model along with IK to 
    # position the arm on top of a cube in unknown position
