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

# if file is in same working directory as data, you can just use the file name
data=pd.read_csv('resdf.csv', index_col=0)

#we have two target values, damage in x and y 
xtarget=data['Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation']
ytarget=data['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation']

target=pd.concat([xtarget,ytarget],axis=1)

#create features dataframe by dropping the target columns
features=data.drop(['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation','Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation'],axis=1)

model = Sequential()
model.add(Dense(100, input_dim=11, activation='relu'))     #CHANGE INPUT_DIM to number of features
# dense layers are fully connected layers, dense(32) means 32 outputs
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2))   #output predicted xdamage and ydamage, both stored in target

#Adam optimizer is used for preliminary results. Other optimizers may be more stable/robust, perhaps less efficient
#Use Keras MSLE metric
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError()) #I think this is MeanSquaredError function

#Adjust number of epochs (iterations over set), validation split, other arguments, and verbosity
#Change input as needed

history = model.fit(features, target, validation_split=0.2, epochs=1000,verbose=0)

model.summary()

#print("model.fit outputs: ",history)
print("NN Average RMSE: ",np.average(history.history['loss']))





#Model evaluation

evaltest = model.evaluate(features,target,batch_size=1)
#print('Accuracy: %.2f' % (accuracy*100))
print(evaltest)



#Plot loss over epochs

plt.plot(history.history['loss'])
plt.show()




