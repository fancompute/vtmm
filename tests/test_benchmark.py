import timeit
import unittest

import numpy as np
import tensorflow as tf

import vtmm
from vtmm.const import C0
from tmm import coh_tmm

from test_tmm import calc_rt_pytmm

NUM = 75

class TestTMM(unittest.TestCase):
    def setUp(self):
        self.omega = tf.linspace(150e12, 250e12, 50) * 2 * np.pi
        self.kx = tf.linspace(0.0, 2 * np.pi * 150e12 / C0, 50)

    def test_benchmark_single(self):
        vec_n = tf.constant([1.0, 3.5, 1.0])
        vec_d = tf.constant([1e-6])
        
        print("Single omega / kx benchmark")
        t1 = timeit.timeit( lambda: vtmm.tmm_rt('p', self.omega[0:1], self.kx[0:1], vec_n, vec_d),  number=NUM )
        print("vtmm: %.4f s" % t1)
        t2 = timeit.timeit( lambda: calc_rt_pytmm('p', self.omega[0:1], self.kx[0:1], vec_n, vec_d), number=NUM )
        print("tmm:  %.4f s" % t2)

    def test_benchmark_small(self):
        vec_n = tf.constant([1.0, 3.5, 1.0])
        vec_d = tf.constant([1e-6])
        
        print("Small stack benchmark")
        t1 = timeit.timeit( lambda: vtmm.tmm_rt('p', self.omega, self.kx, vec_n, vec_d),  number=NUM )
        print("vtmm: %.4f s" % t1)
        t2 = timeit.timeit( lambda: calc_rt_pytmm('p', self.omega, self.kx, vec_n, vec_d), number=NUM )
        print("tmm:  %.4f s" % t2)

    def test_benchmark_medium(self):
        vec_n = tf.constant([1.0, 1.5, 3.5, 1.5, 1.0])
        vec_d = tf.constant([1e-6, 1e-6, 1e-6])

        print("Medium stack benchmark")
        t1 = timeit.timeit( lambda: vtmm.tmm_rt('p', self.omega, self.kx, vec_n, vec_d),  number=NUM )
        print("vtmm: %.4f s" % t1)
        t2 = timeit.timeit( lambda: calc_rt_pytmm('p', self.omega, self.kx, vec_n, vec_d), number=NUM )
        print("tmm:  %.4f s" % t2)

    def test_benchmark_large(self):
        vec_n = tf.constant([1.0, 1.5, 3.5, 1.5, 2.5, 3.0, 1.5, 2, 3, 1.0])
        vec_d = tf.constant([1e-6, 1.33e-6, 1e-6, 1e-6, 2e-6, 1e-5, 1.25e-6, 1e-6])

        print("Large stack benchmark")
        t1 = timeit.timeit( lambda: vtmm.tmm_rt('p', self.omega, self.kx, vec_n, vec_d),  number=NUM )
        print("vtmm: %.4f s" % t1)
        t2 = timeit.timeit( lambda: calc_rt_pytmm('p', self.omega, self.kx, vec_n, vec_d), number=NUM )
        print("tmm:  %.4f s" % t2)


if __name__ == '__main__':
    unittest.main()
