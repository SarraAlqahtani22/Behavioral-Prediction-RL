#LIBRARIES
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.models import Model,load_model
import tensorflow as tf
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.losses import kullback_leibler_divergence
from keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix,accuracy_score



# convert an array of values into a timeseries of 3 previous steps matrix
def create_timeseries(data):
    dataX = []
    dataY = []
    for i in range(3,len(data)):
        if i%25 >= 3:
            a = np.vstack((data[i - 3], data[i - 2],data[i - 1]))
            dataX.append(a)
            dataY.append(data[i])
    return np.array(dataX), np.array(dataY)


#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("Transition_new.npy", allow_pickle=True)
pre = np.asarray(input[:,0])
a1 = np.asarray(input[:,1])
a2 = np.asarray(input[:,2])
a3 = np.asarray(input[:,3])

#flattens the np arrays
pre = np.concatenate(pre).ravel()
pre = np.reshape(pre, (pre.shape[0]//54,54))

data = np.column_stack((pre,a1.T,a2.T,a3.T))
print(data.shape)



#reshapes trainX to be timeseries data with 3 previous timesteps
#LSTM requires time series data, so this reshapes for LSTM purposes
#X has 200000 samples, 3 timestep, 57 features
inputX, inputY = create_timeseries(data)
inputX = inputX.astype('float64')
inputY = inputY.astype('float64')
print(inputX.shape)
print(inputY.shape)


trainX = inputX[:180000]
trainY = inputY[:180000]
valX = inputX[180000:]
valY = inputY[180000:]


valY1 = to_categorical(valY[:,-3])
valY2 = to_categorical(valY[:,-2])
valY3 = to_categorical(valY[:,-1])
trainY1 = to_categorical(trainY[:,-3])
trainY2 = to_categorical(trainY[:,-2])
trainY3 = to_categorical(trainY[:,-1])



#build functional model
visible =Input(shape=(trainX.shape[1],trainX.shape[2]))
hidden1 = LSTM(100, return_sequences=True)(visible)
hidden2 = LSTM(64,return_sequences=True)(hidden1)
#first agent branch
hiddenAgent1 = LSTM(16, name='firstBranch',kernel_regularizer=l2(.01))(hidden2)
agent1 = Dense(valY1.shape[1],activation='softmax',name='agent1classifier',kernel_regularizer=l2(.01))(hiddenAgent1)
#second agent branch
hiddenAgent2 = LSTM(16, name='secondBranch',kernel_regularizer=l2(.01))(hidden2)
agent2 = Dense(valY2.shape[1],activation='softmax',name='agent2classifier',kernel_regularizer=l2(.01))(hiddenAgent2)
#third agent branch
hiddenAgent3 = LSTM(16, name='thirdBranch',kernel_regularizer=l2(.01))(hidden2)
agent3 = Dense(valY3.shape[1],activation='softmax',name='agent3classifier',kernel_regularizer=l2(.01))(hiddenAgent3)


model = Model(inputs=visible,outputs=[agent1,agent2,agent3])

model.compile(optimizer='adam',
              loss={'agent1classifier': 'categorical_crossentropy',
                  'agent2classifier': 'categorical_crossentropy',
                    'agent3classifier': 'categorical_crossentropy'},
              metrics={'agent1classifier': ['acc'],
                       'agent2classifier': ['acc'],
                        'agent3classifier': ['acc']})
print(model.summary())


history = model.fit(trainX,
                    y={'agent1classifier': trainY1,
                       'agent2classifier':trainY2,
                       'agent3classifier':trainY3}, epochs=800, batch_size=5000, verbose=2,
                    validation_data = (valX,
                                       {'agent1classifier': valY1,
                                        'agent2classifier': valY2,
                                        'agent3classifier': valY3}),shuffle=False)

model.save('ActionTransitionNetwork800.keras')


#model = load_model("actionMultiClassNetwork.keras")


np.save("action_transition_history800.npy", history.history, allow_pickle=True)
