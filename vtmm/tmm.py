import tensorflow as tf
from .const import C0
from .fresnel import _r, _t

def tmm_rt(pol, omega, kx, n, d):
    """Calculate the reflection and transmission amplitude for a multi-layer stack

    Parameters:
    pol   - Polarization; should be either 's' or 'p'.
    omega - 1D vector of optical angular frequencies.
    kx    - 1D vector of incident wave vectors.
    n     - 1D vector of layer refractive indices (including first and last).
    d     - 1D vector of layer thicknesses (excluding first and last).

    Returns:
    t , r - 2D tensors of shape [ num kx, num omega ]
    """
    Nd = len(d)
    Nk = len(kx)
    Nn = len(n)
    Nw = len(omega)

    omega = omega.astype('complex128')
    kx = kx.astype('complex128')
    d = d.astype('complex128')
    n = n.astype('complex128')

    assert Nd > 0
    assert Nd == Nn - 2

    n = tf.reshape(n, [-1, 1])
    omega = tf.reshape(omega, [1, -1])
    kx = tf.reshape(kx, [-1, 1, 1])

    k = n * omega / C0
    kz = tf.sqrt(tf.square(k) - tf.square(kx))
    kzn = kz / k # Use normalized kz as a surrogate for cos()

    kzn = tf.reshape(tf.transpose(kzn, perm=[1, 0, 2]), [Nn, -1]) # [Nn, Nk*Nw]
    kz = tf.reshape(tf.transpose(kz, perm=[1, 0, 2]), [Nn, -1]) # [Nn, Nk*Nw]

    # Broadcast
    kz_d = kz[1:-1] * tf.reshape(d, [-1, 1])

    # Flatten
    kz_d = tf.reshape(kz_d, [-1])
    kz_d_conj = tf.math.conj(kz_d)

    # Make diags
    diag_d = tf.stack([tf.math.exp(-1j * kz_d),
                       tf.math.exp( 1j * kz_d_conj)], axis=1)

    D = tf.linalg.diag(diag_d)
    D = tf.reshape(D, [Nd, Nk*Nw, 2, 2])

    # Helpers for constructing TMs
    I = tf.linalg.eye(2, 2, batch_shape=(Nd, Nk*Nw), dtype=d.dtype)
    Ir = tf.roll(I, 1, axis=2)

    # Fresnel coeffs
    rn = _r(pol, n[:-1], n[1:], kzn[:-1], kzn[1:])
    tn = _t(pol, n[:-1], n[1:], kzn[:-1], kzn[1:])
    tn_1 = tf.reshape(tn[1:,:], [Nd, Nk*Nw, 1, 1])
    rn_1 = tf.reshape(rn[1:,:], [Nd, Nk*Nw, 1, 1])

    # All layers and corresponding "trailing" interfaces
    Mn = tf.matmul(D, tf.cast(1/ tn_1 * (I + Ir * rn_1), D.dtype))

    # First interface
    M0 = 1/tf.reshape(tn[0,:],[-1, 1, 1]) * (I[0,:] + Ir[0,:] * tf.reshape(rn[0,:], [-1, 1, 1]))

    M = tf.cast(M0, D.dtype)
    #TODO(ian): see if this can be performed with einsum and if there is a performance benefit
    for i, Mi in enumerate(Mn):
        M = tf.matmul(M, Mi)

    t = tf.reshape(1 / M[:,0,0], [Nk, Nw])
    r = tf.reshape(M[:,1,0] / M[:,0,0], [Nk, Nw])

    return t, r
