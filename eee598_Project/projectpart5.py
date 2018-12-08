# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:51:01 2018

@author: Abhishek
"""

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt

import pandas as pd

 

dataset = pd.read_table( r'C:\Users\Abhishek\Desktop\Machine Learning\Project\Jathar_project_final.txt')

 

feature_names = ['Elec_Facility',

 

                 'Elec_Fans',

 

                'Elec_Cooling',

 

                 'Elec_Heating',

 

                 'Elec_InteriorLights',

 

                 'Elec_InteriorEquipment',

 

                 'Gas_Facility',

 

                 'Gas_Heating',

 

                 'Gas_InteriorEquipment',

 

                'Gas_Water Heater']

 

X = dataset[feature_names].astype(float)

 

y = dataset['Site']

 

# create model

model = Sequential()

model.add(Dense(12, input_dim=10, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

history = model.fit(X, y, validation_split=0.33, epochs=200, batch_size=10, verbose=1)

# list all data in history

print(history.history.keys())

 

# Get current size

fig_size = plt.rcParams["figure.figsize"]

# Set figure width to 12 and height to 9

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size

 

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()