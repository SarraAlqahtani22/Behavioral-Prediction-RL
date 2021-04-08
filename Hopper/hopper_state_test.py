import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf

state_length = 11
action_length = 3

def create_timeseries(x,y,done):
    dataX = []
    dataY = []
    last_done = -1
    for i in range(3,len(data)):
        #if this is the last timestep of an episode, update index of previous done and add data
        if done[i] == 1:
            last_done = i
            dataX.append(np.vstack((x[i - 3], x[i - 2], x[i - 1], x[i])))
            dataY.append(y[i])
        #otherwise, if this is not the last episode AND it has been at least 3 timesteps since the beginning of this episode, add data
        elif done[i] != 1 and i >= last_done - 4:
            dataX.append(np.vstack((x[i - 3], x[i - 2],x[i - 1],x[i])))
            dataY.append(y[i])
    return np.array(dataX),np.array(dataY)


#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../../Datasets/Hopper_DDPG_transition.npy", allow_pickle=True)

#flattens and unpacks the np arrays
pre = np.concatenate(input[:,0]).ravel()
pre = np.reshape(pre, (pre.shape[0]//state_length,state_length))
action = np.concatenate(input[:,1]).ravel()
action = np.reshape(action, (action.shape[0]//action_length,action_length))
post = np.concatenate(input[:,2]).ravel()
post = np.reshape(post, (post.shape[0]//state_length,state_length))
done = np.concatenate(input[:,3]).ravel()
done = np.reshape(done, (done.shape[0]//1,1))

#re-concatenates them
data = np.column_stack((pre,action,post))

inputX = data[:,:action_length+state_length].astype('float64')
inputY = data[:,action_length+state_length:].astype('float64')
inputX,inputY = create_timeseries(inputX,inputY,done)
print(inputX.shape)
print(inputY.shape)


model = load_model('Hopper_State_LSTM.keras')

pred = np.array(model.predict(inputX[0:1]))
print(pred)

