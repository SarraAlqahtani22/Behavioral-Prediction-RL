#LIBRARIES
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.losses import kullback_leibler_divergence
from keras.losses import CategoricalCrossentropy
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix,accuracy_score

def unpack(list):
    list = list.ravel()
    newList = []
    for i in range(list.shape[0]):
        newList.append(list[i][0])
    return np.asarray(newList)


image_width = 84
image_length = 84
action_length = 1

#same results for same model, makes it deterministic
np.random.seed(1234)
tf.random.set_seed(1234)


#reading data
input = np.load("../../Datasets/Pong_DQN_transition.npy", allow_pickle=True)

#flattens and unpacks the np arrays
pre = np.asarray(input[:,0])
post = np.asarray(input[:,2])
pre = unpack(pre)
post = unpack(post)
action = np.concatenate(input[:,1:2]).ravel()
action = np.reshape(action, (action.shape[0]//action_length,action_length))
done = np.concatenate(input[:,3:]).ravel()
done = np.reshape(done, (done.shape[0]//1,1))


inputX = (pre/255)
inputX = np.reshape(inputX,(pre.shape[0],pre.shape[1],pre.shape[2],1))
inputY = post.astype('int32').reshape((post.shape[0],post.shape[1]*post.shape[2]))
print(inputX.shape)
print(inputY.shape)


trainX = inputX[:130000]
trainY = inputY[:130000]
valX = inputX[130000:]
valY = inputY[130000:]



es = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=30)

# design network
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(84, 84,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(200,activation='softmax'))
model.add(Dense(200,activation='softmax'))
model.add(Dense(valY.shape[1],activation='softmax'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# fit network
history = model.fit(trainX, trainY, epochs=1000, batch_size=10000, verbose=2,validation_data = (valX,valY),shuffle=False, callbacks=[es])

model.save('Pong_State_Conv2D.keras')
print(model.summary())

np.save("history_Pong_State_Conv2D.npy", history.history, allow_pickle=True)
