from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
import itertools

########################################################################################################################
# Constants
########################################################################################################################
#Cosine of 60 degrees
C60 = np.cos(1.0472)
#Sine of 60 degrees
S60 = np.sin(1.0472)

########################################################################################################################
# Auxiliary Functions
########################################################################################################################
#Hexagon Number
def chn(n):
    return 1+(6*(0.5*n*(n-1)))

#Hexagon Constructor

def hexagon(cx,cy):
    x = np.array([cx+R*C60, cx+R, cx+R*C60, cx-R*C60, cx-R, cx-R*C60])
    y = np.array([cy+R*S60, cy, cy-R*S60, cy-R*S60, cy, cy+R*S60])
    return x, y

def drawHexagon(x,y):
    for i in range(0,6):
        plt.plot([x[i],x[(i+1)%len(x)]],[y[i],y[(i+1)%len(y)]],'r-',linewidth=2.0)
    del i
    return 1

def permutations(C):
    perms = [p for p in itertools.product(range(-(C - 1), C), repeat=2)]

    for i in list(perms):
        if (i[1] == 0 and i[0] % 2 != 0):
            perms.remove(i)
        if (i[0] % 2 != 0 and abs(i[0]) + abs(i[1]) > C+1):
            perms.remove(i)
        #if (abs(i[0]) + abs(i[1]) > C and C < 4) or abs(i[0]) + abs(i[1]) > C+1:
        if abs(i[0]) + abs(i[1]) > C and abs(i[0]) % 2 == 0:
            perms.remove(i)
    del i
    return perms

def isInHexagon(x,y,cx,cy,R):
    dx = abs(x-cx)/R
    dy = abs(y-cy)/R
    a = 0.25 * S60
    return (dy <= a) and (a*dx + 0.25 * dy <= 0.5 * a)


########################################################################################################################
# Parameters
########################################################################################################################

#Cluster Size (1 to 4 works)
C = 1
#Base Stations (1 per ce'll)
BS = chn(C)
#Cell Radius
R = 100
#Users per Cell
K = 10

########################################################################################################################
# Main Script
########################################################################################################################
plt.figure(1)
idx = permutations(C)

k_x = np.zeros((int(BS), K))
k_y = np.zeros((int(BS), K))
c_x = np.zeros((int(BS), 1))
c_y = np.zeros((int(BS), 1))


i = 0

for j in idx:
    if j[0] % 2 == 0:
        c_x[i] = j[0] * (3 / 2) * R
        c_y[i] = j[1] * S60 * 2 * R
        x, y = hexagon(c_x[i], c_y[i])
        drawHexagon(x, y)
    else:
        c_x[i] = j[0]*(3/2)*R
        c_y[i] = (j[1]*S60*R) + np.sign((j[1])) * (abs(j[1])-1) * S60 * R
        x, y = hexagon(c_x[i], c_y[i])
        drawHexagon(x,y)
    for k in range(0,K):
        k_x[i][k] = np.random.uniform(1, R, 1)
        k_y[i][k] = np.random.uniform(1, R, 1)
        while not(isInHexagon(k_x[i][k], k_y[i][k], c_x[i], c_y[i], R)):
            k_x[i][k] = np.random.uniform(1, R, 1)
            k_y[i][k] = np.random.uniform(1, R, 1)
    i += 1

print(k_x, k_y)
plt.show()