import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

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
    for i in range(0, 6):
        plt.plot([x[i], x[(i+1)%len(x)]], [y[i],y[(i+1)%len(y)]], 'r-', linewidth=2.0)
    del i
    return 1


def drawUser(x,y):
    plt.plot(x, y, 'bs', markersize=2)
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


def genPositions(R):
    x = np.linspace(-R, R, 2*R+1)
    y = np.linspace(int(np.ceil(-R*S60)), int(np.floor(R*S60)), int(np.floor(R*S60) - np.ceil(-R*S60) + 1))
    xg, yg = np.meshgrid(x, y, sparse=False, indexing='ij')
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            if xg[i,j] == 0 and yg[i,j] == 0:
                np.delete(xg,[i, j])
                np.delete(yg,[i, j])
            elif isInHexagon(xg[i,j],yg[i,j],0,0,R) != 1:
                np.delete(xg,[i, j])
                np.delete(yg,[i, j])
    del i, j, x, y
    return xg, yg


def transferPosition(x, y, cx, cy):
    return x+cx, y+cy


def randomPosition(xg,yg):
    i = -1
    j = -1
    while i < 0 or j < 0:
        try:
            i = np.random.randint(0, len(xg)-1, 1)
            j = np.random.randint(0, len(yg)-1, 1)
            tempx = xg[i,j]
            tempy = yg[i,j]
        except IndexError:
            i = -1
            j = -1
    return tempx, tempy


def isInHexagon(x,y,cx,cy,R):
    dx = abs(x-cx)
    dy = abs(y-cy)
    if dx > R or dy > R*S60:
        return False
    #elif R * R * S60 - R * dx - R * S60 * dy >= 0:
        #print(dx, dy, R * R * S60 - R * dx - R * S60 * dy)
        #return False
    elif -2 * np.sqrt(3) * dx + 2*R*np.sqrt(3) - dy < 0:
        return False
    else:
        return True
    # a = 0.25 * S60
    # print(a,dx,dy)
    # return (dy <= a) and (a*dx + 0.25 * dy <= 0.5 * a)


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
K = 50

########################################################################################################################
# Main Script
########################################################################################################################
plt.figure(1)
idx = permutations(C)
xg, yg = genPositions(R)

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
    for k in range(0, K):
        # k_x[i][k] = np.random.uniform(-R + c_x[i], R + c_x[i], 1)
        # k_y[i][k] = np.random.uniform(-R + c_y[i], R + c_y[i], 1)
        # while isInHexagon(k_x[i][k], k_y[i][k], c_x[i], c_y[i], R) != 1:
        #     k_x[i][k] = np.random.uniform(-R + c_x[i], R + c_x[i], 1)
        #     k_y[i][k] = np.random.uniform(-R + c_y[i], R + c_y[i], 1)
        k_x[i, k], k_y[i, k] = randomPosition(xg, yg)
        k_x[i, k], k_y[i, k] = transferPosition(k_x[i, k], k_y[i, k], c_x[i], c_y[i])
        print(k_x[i, k], k_y[i, k])
        print(-2 * np.sqrt(3) * abs(k_x[i, k]) + 2*R*np.sqrt(3) - abs(k_y[i, k]))
        drawUser(k_x[i][k], k_y[i][k])
    i += 1

#print(k_x, k_y)
plt.show()
