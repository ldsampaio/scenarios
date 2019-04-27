import numpy as np
import matplotlib.pyplot as plt
import itertools

########################################################################################################################
# To do List
########################################################################################################################
# Create Pilot Sequence allocation hipermatrix (phi)
# Create Fitness Function
# Implement GA Using fitness function and beta hipermatrix along with phi

########################################################################################################################
# Constants
########################################################################################################################
# Cosine of 60 degrees
C60 = np.cos(1.0472)
# Sine of 60 degrees
S60 = np.sin(1.0472)


########################################################################################################################
# Auxiliary Functions
########################################################################################################################


def hexagon(cx, cy, R):
    x = np.array([cx + R * C60, cx + R, cx + R * C60, cx - R * C60, cx - R, cx - R * C60])
    y = np.array([cy + R * S60, cy, cy - R * S60, cy - R * S60, cy, cy + R * S60])
    return x, y


def drawHexagon(x, y):
    for i in range(0, 6):
        plt.plot([x[i], x[(i + 1) % len(x)]], [y[i], y[(i + 1) % len(y)]], 'r-', linewidth=2.0)
    return 1


def drawUser(x, y):
    plt.plot(x, y, 'bs', markersize=2)
    return 1


def chn(n):
    return 1 + (6 * (0.5 * n * (n - 1)))


def permutations(C):
    perms = [p for p in itertools.product(range(-(C - 1), C), repeat=2)]

    for i in list(perms):
        if i[1] == 0 and i[0] % 2 != 0:
            perms.remove(i)
        if i[0] % 2 != 0 and abs(i[0]) + abs(i[1]) > C + 1:
            perms.remove(i)
        if abs(i[0]) + abs(i[1]) > C and abs(i[0]) % 2 == 0:
            perms.remove(i)

    return perms


def genPositions(R):
    x = np.linspace(-R, R, 2 * R + 1)
    y = np.linspace(int(np.ceil(-R * S60)), int(np.floor(R * S60)), int(np.floor(R * S60) - np.ceil(-R * S60) + 1))
    xg, yg = np.meshgrid(x, y, sparse=False, indexing='ij')
    rx = np.zeros((xg.shape[0], xg.shape[1]), np.float)
    ry = np.zeros((yg.shape[0], yg.shape[1]), np.float)
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            if (xg[i, j] != 0 or yg[i, j] != 0) and isInHexagon(xg[i, j], yg[i, j], 0, 0, R):
                rx[i, j] = float(xg[i, j])
                ry[i, j] = float(yg[i, j])
    del x, y, xg, yg
    return rx, ry


def transferPosition(x, y, cx, cy):
    return x + cx, y + cy


def randomPosition(xg, yg, kx, ky):
    i = -1
    j = -1
    while i < 0 or j < 0:
        try:
            i = np.random.randint(0, len(xg) - 1, 1)
            j = np.random.randint(0, len(yg) - 1, 1)
            tempx = xg[i, j]
            tempy = yg[i, j]
            if tempx == 0 and tempy == 0:
                i = -1
            elif tempx in kx and tempy in ky:
                i = -1
        except IndexError:
            i = -1
    return tempx, tempy


def randomBandwidths(W, Bmax):
    p = np.linspace(1E6, int(Bmax), int(((Bmax - 1E6) / 1E6) + 1))
    r = np.random.choice(p, int(W), replace=False)
    return r


def isInHexagon(x, y, cx, cy, R):
    dx = abs(x - cx)
    dy = abs(y - cy)
    if dx > R or dy > R * S60:
        return False
    elif - np.sqrt(3) * dx + R * np.sqrt(3) - dy < 0:
        return False
    else:
        return True


def calcDistance(kx, ky, cx, cy):
    d = np.zeros((len(kx), len(kx[0]), len(cx)))
    for i in range(0, len(kx)):
        for j in range(0, len(kx[0])):
            for r in range(0, len(cx)):
                d[i, j, r] = np.sqrt((kx[i, j] - cx[r]) ** 2 + (ky[i, j] - cy[r]) ** 2)
    return d


def calcPathLoss(d, d0, F, gamma):
    beta = np.zeros((len(d), len(d[0]), len(d[0][0])))
    for i in range(0, len(d)):
        for j in range(0, len(d[0])):
            for r in range(0, len(d[0][0])):
                if (4 * np.pi * F * (d[i, j, r] ** gamma)) == 0:
                    print(d[i, j, r])
                beta[i, j, r] = (((3E8) / (4 * np.pi * F * d0)) ** 2) * ((d0 / d[i, j, r]) ** gamma)
    return beta


def calcShadowingPathLoss(d, d0, F, gamma, S):
    beta = np.zeros((len(d), len(d[0]), len(d[0][0])))
    tmp = np.random.normal(0, np.sqrt(S), beta.shape)
    for i in range(0, len(d)):
        for j in range(0, len(d[0])):
            for r in range(0, len(d[0][0])):
                if (4 * np.pi * F * (d[i, j, r] ** gamma)) == 0:
                    print(d[i, j, r])
                tmp[i, j, r] = 10 ** (tmp[i, j, r] / 10)
                beta[i, j, r] = ((3E8 / (4 * np.pi * F * d0)) ** 2) * ((d0 / d[i, j, r]) ** gamma) * tmp[i, j, r]
    return beta, tmp


def drawScenario(C, R, K, c_x, c_y, k_x, k_y):
    idx = permutations(C)
    i = 0
    plt.figure()
    for j in idx:
        if j[0] % 2 == 0:
            c_x[i] = j[0] * (3 / 2) * R
            c_y[i] = j[1] * S60 * 2 * R
            x, y = hexagon(c_x[i], c_y[i], R)
            drawHexagon(x, y)
        else:
            c_x[i] = j[0] * (3 / 2) * R
            c_y[i] = (j[1] * S60 * R) + np.sign((j[1])) * (abs(j[1]) - 1) * S60 * R
            x, y = hexagon(c_x[i], c_y[i], R)
            drawHexagon(x, y)
        for k in range(0, K):
            drawUser(k_x[k][i], k_y[k][i])
        i += 1
    return


########################################################################################################################
# Main Function
########################################################################################################################
def create_scenario(C, R, K, Fmin, Fmax, S, gamma, W, Bmax, d0):
    BS = chn(C)
    idx = permutations(C)
    xg, yg = genPositions(R)

    k_x = np.zeros((K, int(BS)))
    k_y = np.zeros((K, int(BS)))
    c_x = np.zeros((int(BS), 1))
    c_y = np.zeros((int(BS), 1))
    beta = np.zeros((int(K), int(BS),  int(BS), int(W)))
    tmp = np.zeros((int(K), int(BS), int(BS), int(W)))

    # Random Frequencies and Bandwidths
    B = randomBandwidths(W, Bmax)
    F = np.random.uniform(Fmin, Fmax, W)
    F = np.floor(F)

    # Random User Positions
    i = 0
    for j in idx:
        if j[0] % 2 == 0:
            c_x[i] = j[0] * (3 / 2) * R
            c_y[i] = j[1] * S60 * 2 * R
        else:
            c_x[i] = j[0] * (3 / 2) * R
            c_y[i] = (j[1] * S60 * R) + np.sign((j[1])) * (abs(j[1]) - 1) * S60 * R
        for k in range(0, K):
            k_x[k, i], k_y[k, i] = randomPosition(xg, yg, k_x, k_y)
            k_x[k, i], k_y[k, i] = transferPosition(k_x[k, i], k_y[k, i], c_x[i], c_y[i])
        i += 1

    # Calculating Distances
    d = calcDistance(k_x, k_y, c_x, c_y)

    # Calculating Path Loss and Shadowing
    for i in range(0, len(B)):
        beta[:, :, :, i], tmp[:, :, :, i] = calcShadowingPathLoss(d, d0, F[i], gamma, S)

    # Calculating Fading (TBD)

    # Returning Scenario Specs
    return k_x, k_y, c_x, c_y, d, tmp, beta
