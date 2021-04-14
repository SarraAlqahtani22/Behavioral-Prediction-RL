import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from tensorflow.python.keras.models import Model, load_model, Sequential
import tensorflow as tf


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
input = np.load("../../Datasets/Pong_DQN_transition.npy", allow_pickle=True)[:90000]

#flattens and unpacks the np arrays
pre = np.asarray(input[:,0])
post = np.asarray(input[:,2])
pre = unpack(pre)
post = unpack(post)
action = np.concatenate(input[:,1:2]).ravel()
action = np.reshape(action, (action.shape[0]//action_length,action_length))
done = np.concatenate(input[:,3:]).ravel()
done = np.reshape(done, (done.shape[0]//1,1))


inputX = (pre/255).reshape((pre.shape[0],pre.shape[1],pre.shape[2],1))
inputY = (post/255).reshape((post.shape[0],post.shape[1],post.shape[2],1))
print(inputX.shape)
print(inputY.shape)


trainX = inputX[:70000]
trainY = inputY[:70000]
valX = inputX[70000:]
valY = inputY[70000:]



es = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=150)

# design network
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(84, 84,1), padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Conv2D(8, (3, 3), activation='relu',padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3),padding='same'))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])

# fit network
history = model.fit(trainX, trainY, epochs=5000, batch_size=1000, verbose=2,validation_data = (valX,valY),shuffle=False, callbacks=[es])

model.save('Pong_State_Conv2D150.keras')
print(model.summary())

np.save("history_Pong_State_Conv2D150.npy", history.history, allow_pickle=True)
