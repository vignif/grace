from grace.utils import RotMat, rotation_matrix_from_vectors
import numpy as np
import unittest

class TestSympy(unittest.TestCase):
    
    def test_rotmat(self):
        R = RotMat(0.5)
        assert R.shape == (2, 2)
        assert R.dtype == np.float64
        assert np.allclose(R, np.array([[0.87758256, -0.47942554],
                                        [0.47942554, 0.87758256]]))

    def test_rotmat_from_vectors(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        R = rotation_matrix_from_vectors(a, b)
        assert R.shape == (3, 3)
        assert R.dtype == np.float64
