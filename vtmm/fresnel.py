from .backend import backend as bd
from .const import C0

def _rs(n1, n2, kzn1, kzn2):
    """Fresnel reflection for s-polarization
    """
    return bd.divide(n1*kzn1 - n2*kzn2, n1*kzn1 + n2*kzn2)

def _ts(n1, n2, kzn1, kzn2):
    """Fresnel transmission for s-polarization
    """
    return bd.divide(2 * n1 * kzn1, n1*kzn1 + n2*kzn2)

def _rp(n1, n2, kzn1, kzn2):
    """Fresnel reflection for p-polarization
    """
    return bd.divide(n2*kzn1 - n1*kzn2, n2*kzn1 + n1*kzn2)

def _tp(n1, n2, kzn1, kzn2):
    """Fresnel transmission for p-polarization
    """
    return bd.divide(2 * n1*kzn1, n2*kzn1 + n1*kzn2)

def _r(pol, n1, n2, kzn1, kzn2):
    """Fresnel reflection
    """
    if pol == 's':
        return _rs(n1, n2, kzn1, kzn2)
    elif pol == 'p':
        return _rp(n1, n2, kzn1, kzn2)
    else:
        raise ValueError('pol must be either s or p')
        
def _t(pol, n1, n2, kzn1, kzn2):
    """Fresnel transmission
    """
    if pol == 's':
        return _ts(n1, n2, kzn1, kzn2)
    elif pol == 'p':
        return _tp(n1, n2, kzn1, kzn2)
    else:
        raise ValueError('pol must be either s or p')
