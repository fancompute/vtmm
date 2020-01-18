import unittest

import numpy as np
import tensorflow as tf

import vtmm
from vtmm.const import C0
from tmm import coh_tmm

TOL = 1e-2

def calc_rt_pytmm(pol, omega, kx, n, d):
    """API-compatible wrapper around pytmm
    """
    vec_omega = omega.numpy()
    vec_lambda = C0/vec_omega*2*np.pi

    vec_n = n.numpy()

    vec_d = d.numpy()
    vec_d = np.append(np.inf, vec_d)
    vec_d = np.append(vec_d, np.inf)

    vec_kx = kx.numpy().reshape([-1,1])
    vec_k0 = 2 * np.pi / vec_lambda.reshape([1,-1])

    vec_theta = np.arcsin(vec_kx / vec_k0)

    r = np.zeros((len(kx), len(omega)), dtype=np.complex64)
    t = np.zeros((len(kx), len(omega)), dtype=np.complex64)

    for i, theta in enumerate(vec_theta):
        for j, lam in enumerate(vec_lambda):
            out = coh_tmm(pol, vec_n, vec_d, theta[j], lam)
            r[i, j] = out['r']
            t[i, j] = out['t']

    t = tf.constant(t)
    r = tf.constant(r)
    
    return tf.constant(t), tf.constant(r)

class TestTMM(unittest.TestCase):
    def setUp(self):
        self.omega = tf.linspace(150e12, 250e12, 50) * 2 * np.pi
        self.kx = tf.linspace(0.0, 2 * np.pi * 150e12 / C0, 51)

    def test_single_p(self):
        vec_n = tf.constant([1.0, 1.5, 1.0])
        vec_d = tf.constant([1e-6])

        tp0, rp0 = vtmm.tmm_rt('p', self.omega, self.kx, vec_n, vec_d)
        tp1, rp1 = calc_rt_pytmm('p', self.omega, self.kx,  vec_n, vec_d)

        self.assertLessEqual(np.linalg.norm(tp0.numpy()-tp1.numpy()), TOL)
        self.assertLessEqual(np.linalg.norm(rp0.numpy()-rp1.numpy()), TOL)

    def test_single_s(self):
        vec_n = tf.constant([1.0, 1.5, 1.0])
        vec_d = tf.constant([1e-6])

        ts0, rs0 = vtmm.tmm_rt('s', self.omega, self.kx, vec_n, vec_d)
        ts1, rs1 = calc_rt_pytmm('s', self.omega, self.kx, vec_n, vec_d)

        self.assertLessEqual(np.linalg.norm(ts0.numpy()-ts1.numpy()), TOL)
        self.assertLessEqual(np.linalg.norm(rs0.numpy()-rs1.numpy()), TOL)

    def test_double_p(self):
        vec_n = tf.constant([1.0, 1.5, 3.5, 1.0])
        vec_d = tf.constant([1e-6, 1.33e-6])

        tp0, rp0 = vtmm.tmm_rt('p', self.omega, self.kx,vec_n, vec_d)
        tp1, rp1 = calc_rt_pytmm('p', self.omega, self.kx, vec_n, vec_d)

        self.assertLessEqual(np.linalg.norm(tp0.numpy()-tp1.numpy()), TOL)
        self.assertLessEqual(np.linalg.norm(rp0.numpy()-rp1.numpy()), TOL)

    def test_double_s(self):
        vec_n = tf.constant([1.0, 1.5, 3.5, 1.0])
        vec_d = tf.constant([1e-6, 1.33e-6])

        ts0, rs0 = vtmm.tmm_rt('s', self.omega, self.kx, vec_n, vec_d)
        ts1, rs1 = calc_rt_pytmm('s', self.omega, self.kx, vec_n, vec_d)

        self.assertLessEqual(np.linalg.norm(ts0.numpy()-ts1.numpy()), TOL)
        self.assertLessEqual(np.linalg.norm(rs0.numpy()-rs1.numpy()), TOL)

if __name__ == '__main__':
    unittest.main()
