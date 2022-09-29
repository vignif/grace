import unittest
import logging
import os
import sys
import numpy as np
import shutil
from grace.utils import GaussianModel

os.path.join(os.path.dirname(__file__), '../grace')
sys.path.append(os.path.join(os.path.dirname(__file__), '../grace'))

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

plt_logger = logging.getLogger("matplotlib")
plt_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)

list_m1 = np.arange(-10, 10, 0.3)
list_m2 = np.arange(-10, 10, 0.1)
param_list = [(m1, m2) for m1 in list_m1 for m2 in list_m2]


class TestSympy(unittest.TestCase):
    def setUp(self):
        log.debug("creating common resources")
        self.path = "./plots"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def tearDown(self):
        shutil.rmtree(self.path)
        log.debug("tearing down common resources")

    def test_gaussian_methods(self):
        log.info(self.id().split('.')[-1])
        G_closed = GaussianModel(1, 2, method="closed")
        G_standard = GaussianModel(1, 2, method="standard")
        self.assertEqual(G_closed, G_standard)

    def test_gaussian_same_means(self):
        log.info(self.id().split('.')[-1])
        G_closed = GaussianModel(1, 1, method="closed")
        G_standard = GaussianModel(1, 1, method="standard")
        self.assertTrue(G_closed.same)
        self.assertTrue(G_standard.same)

    def test_gaussian_different_means(self):
        log.info(self.id().split('.')[-1])
        for p1, p2 in param_list:
            with self.subTest():
                G = GaussianModel(p1, p2)
                self.assertIsInstance(G.value, float)
                self.assertIsInstance(G.itx, float)

    def test_gaussian_plot(self):
        log.info(self.id().split('.')[-1])

        G = GaussianModel(1, 2)
        self.assertIsNone(G.file_path)

        G.plot_once(type="file", path=self.path)
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.isfile(G.file_path))
        print()
