import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())


from utils.encoding import tokenize, TokenizeTransformer
from utils.encoding import AverageWordTokenTransformer


def test_tokenize():
    assert tokenize('hey bro you', {'you'}) == ['hey', 'bro']


def test_tokenize_transform():
    assert list(TokenizeTransformer().transform(['hey bro'])) == [['hey', 'bro']]


def test_average_word_token():
    w = {'a': [1, 0], 'b': [0, 1]}
    r = AverageWordTokenTransformer(w).transform([['a', 'b']])
    assert np.all(r == np.array([[0.5, 0.5]]))
