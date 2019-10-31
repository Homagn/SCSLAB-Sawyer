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
import numpy as np
seed = 7
np.random.seed(seed)
import pickle
#import gym
from sawyer import sawyer
import random
from collections import deque
import copy
import rospy
import csv
import sys

def writelog(reward,done,p1,p2,values1,values2,message):
    fname = 'rewards.csv'
    file1 = open(fname, 'a')
    writer = csv.writer(file1)
    fields1=[reward,done,p1,p2,values1,values2,message]
    writer.writerow(fields1)
    file1.close()
def report_stats(e, episodes, time_t,agent_num_head_train,agent_memory_head,agent_head_history,agent_num_hand_train,agent_memory_hand,agent_hand_history):
    print("episode: {}/{}, score: {}"
          .format(e, episodes, time_t))
    print("Number of time head trained ",agent_num_head_train)
    print("Stored memory length for head images ",len(agent_memory_head))
    try:
        print("Head training history : ",agent_head_history.history['loss'])
    except:
        print("Model not trained yet ")
    print("Number of times hand trained ",agent_num_hand_train)
    print("Stored memory length for hand images ",len(agent_memory_hand))
    try:
        print("Hand training history : ",agent_hand_history.history['loss'])
    except:
        print("Model not trained yet ")
def model_pred_to_robot_acts(action,flag):
    actions=[]
    if(flag==False):
        for i in action[:-1]: #Last one for the switch
            if(i%3==0):
                actions.append(0.0)
            if(i%3==1):
                actions.append(0.05)
            if(i%3==2):
                actions.append(-0.05)
    if(flag==True):
        for i in action[:-1]: #Last one for the switch
            if(i%3==0):
                actions.append(0.0)
            if(i%3==1):
                actions.append(0.05)
            if(i%3==2):
                actions.append(-0.05)
    return actions

class DQNAgent:
    def __init__(self, env, action_size, switches):
        obs = env.reset()
        #env.render()
        #print('initial observation:', obs)

        #action = env.action_space.sample()
        #obs, r, done = env.step(action)
        #print('next observation:', obs)
        #print('reward:', r)
        #print('done:', done)
        #print('info:', info)
        self.state_size = (obs["image"].shape[0])**2
        self.imsize=obs["image"].shape[0]
        print("state size ",self.state_size)
        self.switches=switches
        self.action_size = action_size+switches
        self.memory_head = deque(maxlen=3000)
        self.memory_hand = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.numruns=0
        self.ver=0
        self.cur_mem_head=0
        self.prev_mem_head=0
        self.cur_mem_hand=0
        self.prev_mem_hand=0
        self.view_state="head"
        self.evaluate=True
        self.model_head = self._build_model()
        self.target_model_head = self._build_model()
        
        self.model_hand = self._build_model()
        self.target_model_hand = self._build_model()

        #self.model_head.load_weights("model_weights_head.h5")
        #self.model_hand.load_weights("model_weights_hand.h5")

        self.continuation=True
        if(self.continuation==True):
            self.model_head.load_weights("saved_weights/model_weights_head.h5")
            self.model_hand.load_weights("saved_weights/model_weights_hand.h5")
            if(self.evaluate==False):
                print("Loading the past experience ")
                try:
                    file = open('Experience/experience2.obj', 'r')
                    self.memory_head = pickle.load(file)
                    file = open('Experience/experience2.obj', 'r')
                    self.memory_hand = pickle.load(file)
                except:
                    print("Failed to load past experience. Make sure you have it")
                    #self.memory_head  = []
                    #self.memory_hand = []
                
        #sample=self.model_head.predict([np.zeros((1,1,80,80)),np.zeros((1,7))])
        #print("sample prediction  ",sample)
        #print("sample prediction shape ",sample.shape)
        #sys.exit(0)
        
        self.num_head_train=0
        self.num_hand_train=0
        

        self.head_history=[]
        self.hand_history=[]
        self.TAU=0.01
        # serialize model to JSON
        model_json = self.model_head.to_json() #Model_head and model_hand have the same architecture
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        print("initialized agent and built model")

    def target_train(self,view):
        actor_weights=[]
        actor_target_weights=[]
        if(view=="hand"):
            actor_weights = self.model_hand.get_weights()
            actor_target_weights = self.target_model_hand.get_weights()
        if(view=="head"):
            actor_weights = self.model_head.get_weights()
            actor_target_weights = self.target_model_head.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        if(view=="hand"):
            self.target_model_hand.set_weights(actor_target_weights)
        if(view=="head"):
            self.target_model_head.set_weights(actor_target_weights)
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        image_input= Input(shape=(1, 80, 80))
        joint_input= Input(shape=(7,))
        x=Conv2D(32, (3, 3), input_shape=(1, 80, 80), padding='same', activation='relu', kernel_constraint=maxnorm(3))(image_input)
        x=Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
        x=Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
        x=Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))(x)
        x=MaxPooling2D(pool_size=(2, 2))(x)
        x=Flatten()(x)
        x=Dense(512, activation='relu', kernel_constraint=maxnorm(3))(x)
        x=Dense(100, activation='relu', kernel_constraint=maxnorm(3))(x)

        
        x1 = Dense(30, activation='relu')(x)
        y1= concatenate([x1,joint_input])
        y1 = Dense(10, activation='relu')(y1)
        x12 = Dense(3, activation='linear')(y1)

        x2 = Dense(30, activation='relu')(x)
        y2= concatenate([x2,joint_input])
        y2 = Dense(10, activation='relu')(y2)
        x22 = Dense(3, activation='linear')(y2)

        x3 = Dense(30, activation='relu')(x)
        y3= concatenate([x3,joint_input])
        y3 = Dense(10, activation='relu')(y3)
        x32 = Dense(3, activation='linear')(y3)

        x4 = Dense(30, activation='relu')(x)
        y4= concatenate([x4,joint_input])
        y4 = Dense(10, activation='relu')(y4)
        x42 = Dense(3, activation='linear')(y4)

        x5 = Dense(30, activation='relu')(x)
        y5= concatenate([x5,joint_input])
        y5 = Dense(10, activation='relu')(y5)
        x52 = Dense(3, activation='linear')(y5)

        x6 = Dense(30, activation='relu')(x)
        y6= concatenate([x6,joint_input])
        y6 = Dense(10, activation='relu')(y6)
        x62 = Dense(3, activation='linear')(y6)

        x7 = Dense(30, activation='relu')(x)
        y7= concatenate([x7,joint_input])
        y7 = Dense(10, activation='relu')(y7)
        x72 = Dense(3, activation='linear')(y7)

        x8 = Dense(10, activation='relu')(x)
        y8= concatenate([x8,joint_input])
        y8 = Dense(10, activation='relu')(y8)
        x82 = Dense(2, activation='sigmoid')(y8)

        combined_action = concatenate([x12, x22, x32, x42, x52, x62, x72, x82])
        model = Model(inputs=[image_input,joint_input], outputs=combined_action)
        def huber_loss(a, b, in_keras=True):
            error = a - b
            quadratic_term = error*error / 2
            linear_term = abs(error) - 1/2
            use_linear_term = (abs(error) > 1.0)
            if in_keras:
                # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
                use_linear_term = K.cast(use_linear_term, 'float32')
            return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term
        #model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
        #model.add(Dense(100, activation='relu', kernel_constraint=maxnorm(3)))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.add(Dense(self.action_size, activation='tanh'))

        #model.add(Dense(200, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(50, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.load_weights("model_weights.h5")
        #print("Successfully loaded model_weights")
        #model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        #model.compile(loss='mse',optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0))
        model.compile(loss=huber_loss,optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0))
        return model
    def remember(self, state, action, reward, next_state, done, switched):
        if(self.view_state=="head"):
            self.memory_head.append((state, action, reward, next_state, done, switched))
            self.cur_mem_head+=1
        if(self.view_state=="hand"):
            self.memory_hand.append((state, action, reward, next_state, done, switched))
            self.cur_mem_hand+=1
        self.numruns+=1
        save_file_name_head='Experience/experience_head'+str(self.ver)+'.obj'
        save_file_name_hand='Experience/experience_hand'+str(self.ver)+'.obj'
        if(self.numruns%2000==0 and self.ver<3):#Adds a new experience file for every 2000 runs
            self.ver+=1
        if(self.numruns%100==0): #update memory every 100 runs
            print("Saving experience head")
            exp_file = open(save_file_name_head, 'w')
            pickle.dump(self.memory_head, exp_file)
            exp_file.close()
            print("Saving experience hand")
            exp_file = open(save_file_name_hand, 'w')
            pickle.dump(self.memory_hand, exp_file)
            exp_file.close()
    def act(self, state):
        act_values=[]
        if(self.evaluate==False):
            if ((np.random.rand() <= self.epsilon)or len(self.memory_head)<100 or len(self.memory_hand)<100):
            #if (np.random.rand() <= self.epsilon):
                #act_values= np.random.rand(1,self.action_size)
                print("Random action taken")
                acts=np.random.randint(3,size=7)
                for a in range(len(acts)): #Major bug fix..random actions were always been wrongly taken
                    add=3*a
                    acts[a]+=add
                switches=np.random.randint(2,size=1)
                return np.concatenate((acts,switches)),acts,acts,"random"
        if(self.view_state=="head"):
            act_values = self.model_head.predict([state["image"],state["joints"]])
        if(self.view_state=="hand"):
            act_values = self.model_hand.predict([state["image"],state["joints"]])
        #print("got act values ",act_values)
        a_v=act_values[0][:-self.switches]
        acts=[]
        for i in range(0,len(a_v),3):#3 possible actions for each joint: increase, decrease or remain same
            j=a_v[i:i+3]
            acts.append(np.argmax(j)+i) #Major bug fix
        s_v=act_values[0][-self.switches:]
        for i in range(0,len(s_v),2):#2 possible actions for each switch: change to the other state or remain same
            j=s_v[i:i+2]
            acts.append(np.argmax(j))
        return acts,self.model_head.predict([state["image"],state["joints"]]),self.model_hand.predict([state["image"],state["joints"]]),self.view_state
        #j1,j2,j3,j4,j5,j6,j7=a_v[0:3],a_v[3:6],a_v[6:9],a_v[9:12],a_v[12:15],a_v[15:18],a_v[18:21]
        #return [np.argmax(j1),np.argmax(j2),np.argmax(j3),np.argmax(j4),np.argmax(j5),np.argmax(j6),np.argmax(j7)]  # returns action
    def replay(self, batch_size):
        minibatch=[]
        model=[]
        model_switch=[]
        model_target=[]
        if(self.view_state=="head"):
            minibatch = random.sample(self.memory_head, batch_size)
            model=self.model_head
            model_switch=self.model_hand
            model_target=self.target_model_head

        if(self.view_state=="hand"):
            minibatch = random.sample(self.memory_hand, batch_size)
            model=self.model_hand
            model_switch=self.model_head
            model_target=self.target_model_hand

        states_images=np.zeros((1,1,80,80))
        states_joints=np.zeros((1,7))
        target_fs=np.zeros((1,23)) #7*3 for actions and 2 for switching
        for state, action, reward, next_state, done, switched in minibatch:
            target1 = reward
            target2 = reward
            target3 = reward
            target4 = reward
            target5 = reward
            target6 = reward
            target7 = reward
            target8 = reward
            if not done:
                model_pred=model_target.predict([next_state["image"],next_state["joints"]])[0]
                target1 = reward + self.gamma * np.amax(model_pred[0:3])
                target2 = reward + self.gamma * np.amax(model_pred[3:6])
                target3 = reward + self.gamma * np.amax(model_pred[6:9])
                target4 = reward + self.gamma * np.amax(model_pred[9:12])
                target5 = reward + self.gamma * np.amax(model_pred[12:15])
                target6 = reward + self.gamma * np.amax(model_pred[15:18])
                target7 = reward + self.gamma * np.amax(model_pred[18:21])
                target8 = reward + self.gamma * np.amax(model_pred[21:23])
            target_f =[]
            if(switched==True):
                target_f = model_switch.predict([state["image"],state["joints"]])
            if(switched==False):
                target_f = model.predict([state["image"],state["joints"]])
            target_f[0][action[0]] = target1
            target_f[0][action[1]] = target2
            target_f[0][action[2]] = target3
            target_f[0][action[3]] = target4
            target_f[0][action[4]] = target5
            target_f[0][action[5]] = target6
            target_f[0][action[6]] = target7
            target_f[0][action[7]+21] = target8 #was making a major mistake for this variable responsible for switching
            states_images=np.vstack((states_images,state["image"]))
            states_joints=np.vstack((states_joints,state["joints"]))
            target_fs=np.vstack((target_fs,target_f))
        print("#####################")
        print("Please wait, training model "+self.view_state)
        print("#####################")
        if(self.view_state=="head"):
            self.head_history=self.model_head.fit([states_images[1:],states_joints[1:]], target_fs[1:], epochs=1, verbose=2)#One minibatch update
            self.model_head.save_weights("model_weights_head.h5")
            self.target_model_head.save_weights("target_model_weights_head.h5")
        if(self.view_state=="hand"):
            self.hand_history=self.model_hand.fit([states_images[1:],states_joints[1:]], target_fs[1:], epochs=1, verbose=2)#One minibatch update
            self.model_hand.save_weights("model_weights_hand.h5")
            self.target_model_hand.save_weights("target_model_weights_hand.h5")
        # serialize weights to HDF5
        print("Saved model to disk")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
if __name__ == "__main__":
    env= sawyer()
    #rospy.spin()
    num_joints=7
    outputs=7*3#Increase, decrease by 0.1 or remain same for each of the joints 
    agent = DQNAgent(env,outputs,2)
    agent.evaluate=True
    episodes=10000
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        for time_t in range(20): #Gives 50 tries to the robot to keep moving the arm towards the goal
            state["image"]=np.reshape(state["image"], [1, 1,agent.imsize,agent.imsize])
            action,values1,values2,message = agent.act(state) #batch_size=1, num_channels=1
            actions=model_pred_to_robot_acts(action,agent.evaluate)
            
            switch=action[-1]
            actions.append(switch)
            switched=False
            if(switch==0):
                print("Using head camera ")
                if(agent.view_state=="hand"):
                    switched=True #No experience is stored for switching events
                agent.view_state="head"
            if(switch==1):
                print("Using hand camera ")
                if(agent.view_state=="head"):
                    switched=True
                agent.view_state="hand"
                #switch self.model here
            print("Sending joint inc. actions to robot ",actions)
            next_state, reward, done, performance1, performance2 = env.step(actions)

            #writing the reward history
            writelog(reward,done,performance1,performance2,values1,values2,message)
            state["image"]=np.reshape(state["image"], [1, 1, agent.imsize,agent.imsize])
            next_state["image"]=np.reshape(next_state["image"], [1, 1, agent.imsize,agent.imsize])
            if(agent.evaluate==False):
                agent.remember(state, action, reward, next_state, done, switched)
            # make next_state the new current state for the next frame.
            state = copy.copy(next_state)
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                report_stats(e, episodes, time_t,agent.num_head_train,agent.memory_head,agent.head_history,agent.num_hand_train,agent.memory_hand,agent.hand_history)
                break
        # train the agent with the experience of the episode
        if(agent.evaluate==False):
            if(agent.cur_mem_head-agent.prev_mem_head>32):
                agent.prev_mem_head=copy.copy(agent.cur_mem_head)
                cur_state=copy.copy(agent.view_state)
                agent.view_state="head"
                agent.replay(32)
                agent.target_train("head")
                agent.view_state=copy.copy(cur_state)
                agent.num_head_train+=1
            if(agent.cur_mem_hand-agent.prev_mem_hand>32):
                agent.prev_mem_hand=copy.copy(agent.cur_mem_hand)
                cur_state=copy.copy(agent.view_state)
                agent.view_state="hand"
                agent.replay(32)
                agent.target_train("hand")
                agent.view_state=copy.copy(cur_state)
                agent.num_hand_train+=1
