import numpy as np
import math


def positionOfLine(A, B, C):
    Ax, Ay, Bx, By, X, Y = A[0], A[1], B[0], B[1], C[0], C[1]
    position = np.sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
    if position >= 0:
        return 1
    else:
        return -1


def calAngle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    # cal the angle between side ab and bc
    ba = a - b
    bc = c - b

    t1 = np.dot(ba, bc)
    t2 = np.linalg.norm(ba)
    t3 = np.linalg.norm(bc)

    if np.isnan(t1) or np.isnan(t2) or (t2 * t3) == 0:
        return 0
    if np.isnan(t3):
        return 180

    else:
        cosine_angle = t1 / (t2 * t3)
        if cosine_angle > 1:
            cosine_angle = 1
        if cosine_angle < -1:
            cosine_angle = -1
        angle = np.arccos(cosine_angle)
        p = positionOfLine(a, b, c)
        if p == 1:
            return angle * 180 / np.pi
        else:
            return 360 - angle * 180 / np.pi


# ang = calAngle([0, 0], [1, 0], [1, -1])
# print(ang)



def calDisByeuclidean(a, b):
    dis = math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2)
    dis = math.sqrt(dis)
    return dis
