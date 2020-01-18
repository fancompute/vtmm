import timeit
import unittest

import numpy as np
import tensorflow as tf

import vtmm
from vtmm.const import C0
from tmm import coh_tmm

from test_tmm import calc_rt_pytmm

class TestTMM(unittest.TestCase):
    def setUp(self):
        self.omega = tf.linspace(150e12, 250e12, 50) * 2 * np.pi
        self.kx = tf.linspace(0.0, 2 * np.pi * 150e12 / C0, 51)

    def test_benchmark(self):
        vec_n = tf.constant([1.0, 1.5, 3.5, 1.5, 2.5, 3.0, 1.5, 2, 3, 1.0])
        vec_d = tf.constant([1e-6, 1.33e-6, 1e-6, 1e-6, 2e-6, 1e-5, 1.25e-6, 1e-6])

        print("Running benchmark")
        print("----------------------")
        t1 = timeit.timeit( lambda: vtmm.tmm_rt('p', self.omega, self.kx, vec_n, vec_d),  number=50 )
        print("Tensorflow (vectorized implementation): %.3f" % t1)
        t2 = timeit.timeit( lambda: calc_rt_pytmm('p', self.omega, self.kx, vec_n, vec_d), number=50 )
        print("TMM (unvectorized implementation):      %.3f" % t2)


if __name__ == '__main__':
    unittest.main()
