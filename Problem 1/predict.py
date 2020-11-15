# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:03:27 2020

@author: Lyle
"""

def NN():
    model = Sequential()
    model.add(k.layers.BatchNormalization())
    
    model.add(Dense(128, input_dim=11))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation(tf.keras.activations.selu))
    # model.add(k.layers.Activation(tfa.layers.GELU()))
    # model.add(k.layers.Dropout(.1))
    
    model.add(Dense(256))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation(tf.keras.activations.selu))
    # model.add(k.layers.Activation(tfa.layers.GELU()))
    # model.add(k.layers.Dropout(.1))
    
    model.add(Dense(128))
    model.add(k.layers.BatchNormalization())
    model.add(k.layers.Activation(tf.keras.activations.selu))
    # model.add(k.layers.Activation(tfa.layers.GELU()))
    # model.add(k.layers.Dropout(.1))
    
    model.add(Dense(1, activation='relu'))    
    return model

model=NN()
model.load_weights("Results/Machine 0y .98 scaling20")

machnum=0
data=pd.read_csv('machine'+str(machnum)+'df.csv', index_col=0)