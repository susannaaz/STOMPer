import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import scipy as sc

from scipy.interpolate import CubicSpline
OPTIMIZE = None

def _inv(m):
    result = np.array(map(np.linalg.inv, m.reshape((-1,)+m.shape[-2:])))
    return result.reshape(m.shape)


def _mv(m, v):
    return np.einsum('...ij,...j->...i', m, v, optimize=OPTIMIZE)


def _utmv(u, m, v):
    return np.einsum('...i,...ij,...j', u, m, v, optimize=OPTIMIZE)


def _mtv(m, v):
    return np.einsum('...ji,...j->...i', m, v, optimize=OPTIMIZE)


def _mm(m, n):
    return np.einsum('...ij,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mtm(m, n):
    return np.einsum('...ji,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mmv(m, w, v):
    return np.einsum('...ij,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mtmv(m, w, v):
    return np.einsum('...ji,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mmm(m, w, n):
    return np.einsum('...ij,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)


def _mtmm(m, w, n):
    return np.einsum('...ji,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)

def prepare_t(t):
    if t.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    dt = np.diff(t)
    #check for negative numbers
    if np.any(dt <= 0):
        tindx = (np.where(dt<=0))[0] # indices where dt <= 0
        print("`t` is not an increasing sequence.")
        print("Negative dt indices: ",tindx)
        raise ValueError("`t` must be strictly increasing sequence.")
    return t

class lsqCubicSpline:
    def __init__(self, t, bc_type='clamped', extrapolate=None):
        self.t = prepare_t(t)
        self.cs = CubicSpline(t, np.eye(t.size), bc_type = 'clamped')
        self.bc_type = bc_type
        self.extrapolate = extrapolate

    def plot_basis(self):
        for i in self.cs(np.linspace(self.t.min(), self.t.max(), 10* self.t.size)).T:
            plt.plot(i)
            print(i.max())
        plt.show()

    def lsqfit(self, x, y):
        """
        Return least square fit result c and y_fit                                                                                                                                                        
        """
        A = self.cs(x)
        AtA =_mtm(A, A)
        Atb = _mv(A.T, y)
        ####                                                                                                                                                                                
        # seems to the same case for me 
        #c = np.linalg.lstsq(A, y_data)[0]                                                                                                                                                                
        ####                                                                                                                                                                      
        c = np.linalg.solve(AtA, Atb)
        cs = CubicSpline(self.t, c, bc_type = self.bc_type, extrapolate=self.extrapolate)
        return c, cs

# Use w/:                                                                                                                                                                                         
#from spline_fit import *
#spl = lsqCubicSpline(x)
#c, cs = spl.lsqfit(x, y)
#y_spline = cs(x_new)
