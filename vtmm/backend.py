try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

class Backend(object):

    def __repr__(self):
        return self.__class__.__name__

class NumpyBackend(Backend):
    divide = staticmethod(np.divide)
    reshape = staticmethod(np.reshape)
    sqrt = staticmethod(np.sqrt)
    square = staticmethod(np.square)
    transpose = staticmethod(np.transpose)
    stack = staticmethod(np.stack)
    exp = staticmethod(np.exp)
    matmul = staticmethod(np.matmul)
    cast = staticmethod(np.ndarray.astype)
    roll = staticmethod(np.roll)
    constant = staticmethod(np.array)
    diag = staticmethod(np.diag)
    eye = staticmethod(np.eye)
    linspace = staticmethod(np.linspace)
    complex = staticmethod(np.complex)

class TensorflowBackend(Backend):
    divide = staticmethod(tf.math.divide)
    reshape = staticmethod(tf.reshape)
    sqrt = staticmethod(tf.sqrt)
    square = staticmethod(tf.square)
    transpose = staticmethod(tf.transpose)
    stack = staticmethod(tf.stack)
    exp = staticmethod(tf.math.exp)
    matmul = staticmethod(tf.matmul)
    cast = staticmethod(tf.cast)
    roll = staticmethod(tf.roll)
    constant = staticmethod(tf.constant)
    diag = staticmethod(tf.linalg.diag) 
    eye = staticmethod(tf.linalg.eye)
    linspace = staticmethod(tf.linspace)
    complex = staticmethod(tf.complex)

backend = TensorflowBackend()

def set_backend(name: str):
    if name == 'tensorflow' and not TF_AVAILABLE:
        raise ValueError("Tensor Flow backend is not available")

    if name == 'numpy' and not NP_AVAILABLE:
        raise ValueError("Numpy backend is not available")

    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'tensorflow':
        backend.__class__ = TensorflowBackend
    else:
        raise ValueError(f"unknown backend '{name}'")
