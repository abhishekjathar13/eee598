# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:22:01 2018

@author: Abhishek
"""
import pandas

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

 

 

from sklearn import decomposition

 

np.random.seed(5)

 

centers = [[1, 1], [-1, -1], [1, -1]]

 

# Reading data from the file

url     = r'C:\Users\Abhishek\Desktop\Machine Learning\Project\Jathar_project_final.txt'

names   = ['Date_Time',

           'Elec_Facility',

           'Elec_Fans',

           'Elec_Cooling',

           'Elec_Heating',

           'Elec_InteriorLights',

           'Elec_InteriorEquipment',

           'Gas_Facility',

           'Gas_Heating',

           'Gas_InteriorEquipment',

           'Gas_WaterHeater',

           'Site',

           'Randomise']

data    = pandas.read_table(url, names=names, header=0 )

data_X  = data.drop(['Date_Time','Site','Randomise'], axis=1)

data_y  = data.drop(['Date_Time',

                     'Elec_Facility',

                     'Elec_Fans',

                     'Elec_Cooling',

                     'Elec_Heating',

                     'Elec_InteriorLights',

                     'Elec_InteriorEquipment',

                     'Gas_Facility',

                     'Gas_Heating',

                     'Gas_InteriorEquipment',

                     'Gas_WaterHeater',

                     'Randomise'], axis=1)

 

X        = np.array(data_X, dtype=np.float)

y        = np.array(data_y, dtype=np.float)

y        = y.ravel()

 

fig = plt.figure(1, figsize=(4, 3))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, 3.95, 4], elev=48, azim=134)

 

plt.cla()

pca = decomposition.PCA(n_components=3)

pca.fit(X)

X = pca.transform(X)

 

for name, label in [('1', 1),

                    ('2', 2),

                    ('3', 3)]:

    ax.text3D(X[y == label, 0].mean(),

              X[y == label, 1].mean() + 1.5,

              X[y == label, 2].mean(), name,

              horizontalalignment='center',

              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

   

# Reorder the labels to have colors matching the cluster results

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,

           edgecolor='k')

 

ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])

 

 

plt.figure()

colors = ['navy', 'turquoise', 'darkorange', 'red', 'pink', 'yellow']

lw = 2

target_names= ['1', '2', '3', '4', '5', '6']

for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5], target_names):

    plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('PCA 2D')

 

plt.show()
