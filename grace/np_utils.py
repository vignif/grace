import numpy as np
from math import atan2, sin, cos

import quaternion as qt


def inverse_quaternion(q):
    return qt.quaternion(q.w, -q.x, -q.y, -q.z)


def angle_between_vectors(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def RotMat(rad):
    # theta = np.radians(rad)
    # expects input already to be radians
    c, s = np.cos(rad), np.sin(rad)
    return np.array(((c, -s), (s, c)))


def delta_angles(source, target):
    return atan2(sin(target - source), cos(target - source))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
        return v


def q_from_matrix(M):
    m00 = M[0][0]
    m01 = M[0][1]
    m02 = M[0][2]
    m10 = M[1][0]
    m11 = M[1][1]
    m12 = M[1][2]
    m20 = M[2][0]
    m21 = M[2][1]
    m22 = M[2][2]

    if (m22 < 0):
        if (m00 > m11):
            t = 1 + m00 - m11 - m22
            q = [t, m01+m10, m20+m02, m12-m21]

        else:
            t = 1 - m00 + m11 - m22
            q = [m01+m10, t, m12+m21, m20-m02]

    else:
        if (m00 < -m11):
            t = 1 - m00 - m11 + m22
            q = [m20+m02, m12+m21, t, m01-m10]

        else:
            t = 1 + m00 + m11 + m22
            q = [m12-m21, m20-m02, m01-m10, t]

    return normalize([q[0], q[1], -q[2], q[3]])
