# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:14:24 2020

@author: Lyle
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
# if file is in same working directory as data, you can just use the file name
data=pd.read_csv('machine0df.csv', index_col=0)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data.values)
data = pd.DataFrame(x_scaled, columns=data.columns,index=data.index.values)
#we have two target values, damage in x and y 
# xtarget=data['Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation']
# ytarget=data['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation']

xtarget=data[data.columns[1]]
ytarget=data[data.columns[11]]
target=pd.concat([xtarget,ytarget],axis=1)
target=target*20
#create features dataframe by dropping the target columns
# features=data.drop(['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation','Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation'],axis=1)
features=data.drop(data.columns[11],axis=1)
features=features.drop(features.columns[1],axis=1)

model = Sequential()
model.add(k.layers.BatchNormalization())
model.add(Dense(128, input_dim=11, activation='selu'))     #CHANGE INPUT_DIM to number of features
# dense layers are fully connected layers, dense(32) means 32 outputs
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(2, activation='relu'))    #output predicted xdamage and ydamage, both stored in target

#Adam optimizer is used for preliminary results. Other optimizers may be more stable/robust, perhaps less efficient
#Use Keras MSLE metric
# model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError()) #I think this is MeanSquaredError function


def r2_tf(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.math.square(y_true - y_pred),0) 
    SS_tot = tf.reduce_sum(tf.math.square(y_true -  tf.reduce_mean(y_true,0)),0) 
    return (1-SS_res/(SS_tot + tf.keras.backend.epsilon()))
# model.compile(optimizer='adam',loss='mse')
# model.compile(optimizer='adam',loss=custom_loss)
#Adjust number of epochs (iterations over set), validation split, other arguments, and verbosity
#Change input as needed

# history = model.fit(features, target, validation_split=0.2, epochs=100,verbose=0)

batch_size=2500
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

train_dataset=tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_dataset=train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(batch_size)



optimizer = tf.keras.optimizers.Adam(1e-4)

trainloss=[]
valloss=[]
R21=[]
R22=[]
for epoch in range(1000):
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value =tf.keras.losses.MSE(y_batch_train,logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        trainloss.append(np.mean(loss_value))

        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        val_loss=tf.keras.losses.MSE(y_batch_val,val_logits)
        R2val=r2_tf(y_batch_val,tf.cast(val_logits,tf.float64))
        R21.append(R2val[0])
        R21.append(R2val[1])
        print(R2val)
        valloss.append(np.mean(val_loss))
        if step % 200 == 0:
            print(
                "Training loss at step %d: %.4f" % (step, float(trainloss[-1]))
            )
            print("Val loss at step %d: %.4f" % (step, float(valloss[-1])))






model.summary()

#print("model.fit outputs: ",history)
#print("NN Average RMSE: ",np.average(history.history['loss']))

# print("NN Average RMSE over last 50 points: ",np.average(history.history['val_loss'][-50:]))




#Model evaluation

# evaltest = model.evaluate(features,target,batch_size=1)
#print('Accuracy: %.2f' % (accuracy*100))
#print(evaltest)



#Plot loss over epochs

plt.plot(valloss[10:],label= 'Test loss')
plt.plot(trainloss[10:],label= 'Train loss')
plt.plot(R21[10:],label= 'R21')
plt.plot(R22[10:],label= 'R22')
plt.legend()
plt.show()




