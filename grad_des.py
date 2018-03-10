# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:52:01 2018

@author: HatemZam
"""

data = [[1,   1.5],
        [2,   3],
        [3,   2.5],
        [4,   4],
        [4.5, 5.5],
        [5,   5],
        [5.5,  6.8],
        [6,    6]]

## print(data[1][0])
import matplotlib.pyplot as plt

#plt.grid()
#plt.axis([0,6,0,6])
#for i in range(len(data)) :
#    plt.scatter([data[i][0]], [data[i][1]], c='b')

# Initialize theta0 & theta1
init_theta0 = 0.5
init_theta1 = 1
hyp = []

# Claculate hypothesis function
def hypoth(init_theta0, init_theta1, x_features):
    global hyp
    for i in range(len(x_features)) :
        hyp.append(init_theta0 + init_theta1 * x_features[i][0])

alpha = 0.01
m = len(data)

def grad_decse():
    global init_theta0, init_theta1
    for j in range(0, 20):
        for i in range(0, m):
            init_theta0 = init_theta0 - alpha * (1/m) * (hyp[i] - data[i][1])
            init_theta1 = init_theta1 - alpha * (1/m) * (hyp[i] - data[i][1]) * data[i][0]

hypoth(init_theta0, init_theta1, data)
grad_decse()

# print Thetas
print("Theta 0: %s" % str(init_theta0))
print("Theta 1: %s" % str(init_theta1))

# create an array with specific range
import numpy as np
xrange = np.arange(0, 10, 1)
ypred = []

# Calculate the y-predictions with the new (theta0, theta1)
for i in range(len(xrange)):
    global ypred
    ypred.append(init_theta0 + init_theta1 * xrange[i])

# Plot the data with the fitted line of predictions
for i in range(len(data)) :
    plt.scatter([data[i][0]], [data[i][1]], c='b')
plt.plot(ypred, c='r')
plt.show()







