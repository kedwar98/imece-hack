# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:04:25 2020

@author: Kristen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
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
model.add(Dense(2, activation='relu'))   #output predicted xdamage and ydamage, both stored in target

#Adam optimizer is used for preliminary results. Other optimizers may be more stable/robust, perhaps less efficient
#Use Keras MSLE metric
# model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError()) #I think this is MeanSquaredError function

def custom_loss(labels, predictions):
    # result = tfa.metrics.RSquare()
    # result.update_state(y_true, y_pred)
    # return result
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels, axis=0)))
    R2 = 1. - tf.math.divide(unexplained_error, total_error)
    return -R2

def r2_tf(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.math.square(y_true - y_pred),0) 
    SS_tot = tf.reduce_sum(tf.math.square(y_true -  tf.reduce_mean(y_true,0)),0) 
    return (1-SS_res/(SS_tot + tf.keras.backend.epsilon()))

# model.compile(optimizer='adam',loss='mse')
model.compile(optimizer='adam',loss=custom_loss)
#Adjust number of epochs (iterations over set), validation split, other arguments, and verbosity
#Change input as needed

history = model.fit(features, target, validation_split=0.2, epochs=100,verbose=0)







model.summary()

#print("model.fit outputs: ",history)
#print("NN Average RMSE: ",np.average(history.history['loss']))

print("NN Average RMSE over last 50 points: ",np.average(history.history['val_loss'][-50:]))




#Model evaluation

evaltest = model.evaluate(features,target,batch_size=1)
#print('Accuracy: %.2f' % (accuracy*100))
#print(evaltest)



#Plot loss over epochs

plt.plot(history.history['val_loss'][10:],label= 'Test loss')
plt.plot(history.history['loss'][10:],label= 'Train loss')
plt.show()




