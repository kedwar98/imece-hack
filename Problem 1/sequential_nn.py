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

# if file is in same working directory as data, you can just use the file name
data=pd.read_csv('resdf.csv', index_col=0)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data.values)
data = pd.DataFrame(x_scaled, columns=data.columns,index=data.index.values)
#we have two target values, damage in x and y 
xtarget=data['Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation']
ytarget=data['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation']

target=pd.concat([xtarget,ytarget],axis=1)

#create features dataframe by dropping the target columns
features=data.drop(['Machines > Bridgeport Mill 1 > Spindle > Y-Radial > Damage Accumulation','Machines > Bridgeport Mill 1 > Spindle > X-Axial > Damage Accumulation'],axis=1)
model = Sequential()
model.add(k.layers.BatchNormalization())
model.add(Dense(128, input_dim=11, activation='selu'))     #CHANGE INPUT_DIM to number of features
# dense layers are fully connected layers, dense(32) means 32 outputs
model.add(Dense(256, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(2))   #output predicted xdamage and ydamage, both stored in target

#Adam optimizer is used for preliminary results. Other optimizers may be more stable/robust, perhaps less efficient
#Use Keras MSLE metric
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError()) #I think this is MeanSquaredError function

#Adjust number of epochs (iterations over set), validation split, other arguments, and verbosity
#Change input as needed

history = model.fit(features, target, validation_split=0.2, epochs=1000,verbose=0)


model.summary()

#print("model.fit outputs: ",history)
#print("NN Average RMSE: ",np.average(history.history['loss']))

print("NN Average RMSE over last 50 points: ",np.average(history.history['val_loss'][-50:]))




#Model evaluation

evaltest = model.evaluate(features,target,batch_size=1)
#print('Accuracy: %.2f' % (accuracy*100))
#print(evaltest)



#Plot loss over epochs

plt.plot(history.history['val_loss'],label= 'Test loss')
plt.plot(history.history['loss'],label= 'Train loss')
plt.show()




