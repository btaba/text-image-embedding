"""

References:

- https://web.stanford.edu/~lmackey/stats306b/doc/stats306b-spring14-lecture13_scribed.pdf

"""

import numpy as np
from scipy import linalg


def _center(X, Y, Xmean, Ymean):
    X, Y = X.copy(), Y.copy()
    X = X - Xmean
    Y = Y - Ymean
    return X, Y


class CCA:

    def __init__(self):
        pass

    def fit(self, X, Y, n_components, regfactor, center=False, regscaled=False):

        N = X.shape[0]
        assert Y.shape[0] == N

        self._center = center
        if center:
            self._Xmean = X.mean(axis=0)
            self._Ymean = Y.mean(axis=0)
            X, Y = _center(X, Y, self._Xmean, self._Ymean)

        Cxy = X.T.dot(Y)
        Cyx = Y.T.dot(X)
        Cxx = X.T.dot(X)
        Cyy = Y.T.dot(Y)

        # regularize
        if regscaled:
            xscale = linalg.eigh((Cxx.T + Cxx) / 2)[0][-1]
            yscale = linalg.eigh((Cyy.T + Cyy) / 2)[0][-1]
            Cxx += np.eye(Cxx.shape[0]) * xscale * regfactor
            Cyy += np.eye(Cyy.shape[0]) * yscale * regfactor
        else:
            Cxx += regfactor * np.eye(Cxx.shape[0])
            Cyy += regfactor * np.eye(Cyy.shape[0])

        # construct eigenvalue problem
        rhs = linalg.block_diag(Cxx, Cyy)
        lhs = np.zeros(rhs.shape)
        lhs[:Cxy.shape[0], Cyx.shape[1]:] = Cxy
        lhs[Cxy.shape[0]:, :Cyx.shape[1]] = Cyx

        max_components = lhs.shape[0]
        assert n_components <= max_components

        # symmetrize
        rhs = (rhs + rhs.T) / 2.0
        lhs = (lhs + lhs.T) / 2.0

        r, Ev = linalg.eigh(
            lhs, rhs, eigvals=(max_components - n_components, max_components - 1))

        U, V = Ev[:X.shape[1]], Ev[X.shape[1]:]

        self.U = U
        self.V = V
        self.r = r

        return self

    def predict(self, X, Y, scale_by_eigs=False, norm=False):

        if self._center:
            X, Y = _center(X, Y, self._Xmean, self._Ymean)

        X_c, Y_c = X.dot(self.U), Y.dot(self.V)
        
        if scale_by_eigs:
            X_c, Y_c = X_c * self.r, Y_c * self.r

        if norm:
            X_c, Y_c = X_c / np.sum(np.square(X_c), axis=0), Y_c / np.sum(np.square(Y_c), axis=0)

        return X_c, Y_c
