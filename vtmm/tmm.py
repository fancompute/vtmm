from .backend import backend as bd
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

    assert Nd > 0
    assert Nd == Nn - 2

    n = bd.reshape(n, [-1, 1])
    omega = bd.reshape(omega, [1, -1])
    kx = bd.reshape(kx, [-1, 1, 1])

    k = n * omega / C0
    kz = bd.sqrt(bd.square(k) - bd.square(kx))
    kzn = kz/k # Use normalized kz as a surrogate for cos()

    kzn = bd.reshape(bd.transpose(kzn, [1, 0, 2]), [Nn, -1]) # [Nn, Nk*Nw]
    kz = bd.reshape(bd.transpose(kz, [1, 0, 2]), [Nn, -1]) # [Nn, Nk*Nw]

    # Broadcast
    kz_d = kz[1:-1] * bd.reshape(d, [-1, 1])

    # Flatten
    kz_d = bd.reshape(kz_d, [-1])

    # Make diags
    zz = bd.constant(0.0, dtype=d.dtype)
    diag_d = bd.stack([bd.exp(bd.complex(zz, -kz_d)),
                       bd.exp(bd.complex(zz,  kz_d))], axis=1)

    D = bd.diag(diag_d)
    D = bd.reshape(D, [Nd, Nk*Nw, 2, 2])

    # Helpers for constructing TMs
    I = bd.eye(2, 2, batch_shape=(Nd, Nk*Nw), dtype=d.dtype)
    Ir = bd.roll(I, 1, axis=2)

    # Fresnel coeffs
    rn = _r(pol, n[:-1], n[1:], kzn[:-1], kzn[1:])
    tn = _t(pol, n[:-1], n[1:], kzn[:-1], kzn[1:])
    tn_1 = bd.reshape(tn[1:,:], [Nd, Nk*Nw, 1, 1])
    rn_1 = bd.reshape(rn[1:,:], [Nd, Nk*Nw, 1, 1])

    # All layers and corresponding "trailing" interfaces
    Mn = bd.matmul(D, bd.cast(1/ tn_1 * (I + Ir * rn_1), D.dtype))

    # First interface
    M0 = 1/bd.reshape(tn[0,:],[-1, 1, 1]) * (I[0,:] + Ir[0,:] * bd.reshape(rn[0,:], [-1, 1, 1]))

    M = bd.cast(M0, D.dtype)
    #TODO(ian): see if this can be performed with einsum and if there is a performance benefit
    for i, Mi in enumerate(Mn):
        M = bd.matmul(M, Mi)

    t = bd.reshape(1 / M[:,0,0], [Nk, Nw])
    r = bd.reshape(M[:,1,0] / M[:,0,0], [Nk, Nw])

    return t, r
