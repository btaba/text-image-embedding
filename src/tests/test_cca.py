import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())

from models.cca import CCA


def test_cca():
    X = np.eye(3)
    Y = np.eye(3)

    c = CCA().fit(X, Y, 2, 0)
    X_c, Y_c = c.predict(X, Y)

    assert np.all(X_c == Y_c)


def test_cca2():
    X = np.array([[1., 2, 0], [2, 1, 1], [0, 1, 0]])
    Y = np.array([[1., 0], [1, 0], [0, 1]])

    c = CCA().fit(X, Y, 2, 0)
    X_c, Y_c = c.predict(X, Y)
    assert np.allclose(np.diag(c.U.T.dot(X.T.dot(X)).dot(c.U)), 0.5)
    assert np.allclose(np.diag(c.V.T.dot(Y.T.dot(Y)).dot(c.V)), 0.5)
