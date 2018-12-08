# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:49:36 2018

@author: Abhishek
"""

import pandas as pd

 

import matplotlib.pyplot as plt

 

import numpy as np

 

df       = pd.read_table( r'C:\Users\Abhishek\Desktop\Machine Learning\Project\Jathar_project_final.txt')

data_y   = df['Site']

y        = np.array(data_y, dtype=np.float)

 

 

# Get current size

fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]

print ("Current size:", fig_size)

# Set figure width to 12 and height to 9

fig_size[0] = 12

fig_size[1] = 9

plt.rcParams["figure.figsize"] = fig_size

 

#Electricity consumption vs Site

plt.scatter(df['Elec_Facility'], df['Site'])

plt.show()

 

#Gas consumption vs Site

plt.scatter(df['Gas_Facility'], df['Site'])

plt.show()

 

# create histogram for numeric data

df.hist()

plt.show()