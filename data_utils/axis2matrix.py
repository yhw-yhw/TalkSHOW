import numpy as np
import math
import scipy.linalg as linalg


def rotate_mat(axis, radian):

    a = np.cross(np.eye(3), axis / linalg.norm(axis) * radian)

    rot_matrix = linalg.expm(a)

    return rot_matrix

def aaa2mat(axis, sin, cos):
    i = np.eye(3)
    nnt = np.dot(axis.T, axis)
    s = np.asarray([[0, -axis[0,2], axis[0,1]],
                    [axis[0,2], 0, -axis[0,0]],
                    [-axis[0,1], axis[0,0], 0]])
    r = cos * i + (1-cos)*nnt +sin * s
    return r

rand_axis = np.asarray([[1,0,0]])
#旋转角度
r = math.pi/2
#返回旋转矩阵
rot_matrix = rotate_mat(rand_axis, r)
r2 = aaa2mat(rand_axis, np.sin(r), np.cos(r))
print(rot_matrix)