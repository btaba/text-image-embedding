"""
Word and image encoding utility functions
"""
import numpy as np

from functools import partial
from itertools import islice, chain
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from keras.preprocessing import image
from keras.applications import vgg19, inception_resnet_v2, nasnet
from keras.models import Model

from utils.data_utils import open_dataset, stream_json


STOPWORDS_SET = set(stopwords.words('english'))
EMBEDDING_PATHS = open_dataset('word-embeddings')


def load_fasttext_vecs(f='wiki-news-300d-1M-subword.vec.json.txt',
                       top_n=100000):
    
    word_vecs = {}
    for idx, line in enumerate(stream_json(EMBEDDING_PATHS[f])):
        if idx > top_n:
            break
        word_vecs.update(line)

    return word_vecs


def load_numberbatch_vecs(f='numberbatch-en-17.06.json.txt'):
    word_vecs = {}
    for line in stream_json(EMBEDDING_PATHS[f]):
        word_vecs.update(line)
    return word_vecs


def tokenize(sentence, stopwords_set):
    tokens = word_tokenize(sentence)
    return [t for t in tokens if t not in stopwords_set]


def _tokenize_map(sentences, **kwargs):
    func = partial(tokenize, **kwargs)
    return map(func, sentences)


class TokenizeTransformer(FunctionTransformer):
    """
    Apply tokenizer to input text
    """

    def __init__(self, stopwords_set=STOPWORDS_SET):
        kw_args = {
            'stopwords_set': stopwords_set,
        }
        super().__init__(
            _tokenize_map, validate=False,
            kw_args=kw_args)

    def __repr__(self):
        return '<TokenizeTransformer>'


def _average_vecs(X, word_vecs):
    vecs = []
    vec_length = len(next(iter(word_vecs.values())))

    for x in X:
        vec = []

        for token in x:

            if token in word_vecs:
                vec.append(word_vecs[token])

        if len(vec) == 0:
            vec.append(np.zeros(vec_length))

        vecs.append(np.mean(vec, axis=0))

    return np.array(vecs)


class AverageWordTokenTransformer(FunctionTransformer):
    """
    Average word vectors for list of tokens
    """

    def __init__(self, word_vec_dict):
        kw_args = {
            'word_vecs': word_vec_dict,
        }
        super().__init__(
            _average_vecs, validate=False,
            kw_args=kw_args)

    def __repr__(self):
        return '<AverageWordTokenTransformer>'


def fasttext_vectorizer(stopwords_set=STOPWORDS_SET):
    vectorizer = Pipeline([
        ('tokenizer', TokenizeTransformer(stopwords_set=stopwords_set)),
        ('vectorizer', AverageWordTokenTransformer(load_fasttext_vecs()))])
    return vectorizer


def numberbatch_vectorizer(stopwords_set=STOPWORDS_SET):
    vectorizer = Pipeline([
        ('tokenizer', TokenizeTransformer(stopwords_set=stopwords_set)),
        ('vectorizer', AverageWordTokenTransformer(load_numberbatch_vecs()))])
    return vectorizer


def word2vec_vectorizer(stopwords_set=STOPWORDS_SET):
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(
        str(EMBEDDING_PATHS['GoogleNews-vectors-negative300.bin']), binary=True)
    d = {w: model.wv[w] / np.sum(np.square(model.wv[w])) for w in model.vocab}
    vectorizer = Pipeline([
        ('tokenizer', TokenizeTransformer(stopwords_set=stopwords_set)),
        ('vectorizer', AverageWordTokenTransformer(d))])
    return vectorizer


def batch(iterable, size=128):
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, size - 1)))


def get_image_model(module, network_func, layer, imsize):
    model = network_func(include_top=True, weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[layer].output)

    def imread(image_path):
        im = image.load_img(image_path, target_size=imsize)
        im = image.img_to_array(im)
        im = module.preprocess_input(im)
        return im

    def predict_stream(strm):
        for ims in batch(strm):
            yield from model.predict(np.array(ims))

    return predict_stream, imread


def get_vgg19():
    # layer [-3] is fc1, [-2] is fc2
    return get_image_model(vgg19, vgg19.VGG19, -3, (224, 224))


def get_nasnetlarge():
    return get_image_model(nasnet, nasnet.NASNetLarge, -2, (331, 331))


def get_inceptionresnetv2():
    return get_image_model(inception_resnet_v2,
                           inception_resnet_v2.InceptionResNetV2, -2, (299, 299))

