"""
Generate embeddings from `image_tagger_representations` in `encders.py`
"""

import click
import numpy as np
from cca_generate import save
from utils.data_utils import stream_json, BASE_PATH


def load_X_Y(dataset, split, encoding_name):
    json_file = BASE_PATH / dataset / encoding_name /\
        ('%s-tagencoded-captions-and-images.json' % split)

    X = [i['x_image'] for i in stream_json(json_file)]
    Y = [i['x_text'] for i in stream_json(json_file)]
    return np.array(X), np.array(Y)


@click.command()
@click.argument('dataset')
@click.argument('encoding_name')
def text_encoder_generate(dataset, encoding_name):
    # simply save the encodings as numpy ndarrays
    #  since both the image and text representations
    #  are text representations, thus share the same dimensions
    for split in ['train', 'validation', 'test']:
        print('Saving {} embeddings'.format(split))
        X_c, Y_c = load_X_Y(dataset, split, encoding_name)
        save(X_c, Y_c, split, None, dataset, encoding_name, 'textencoder')


if __name__ == '__main__':
    text_encoder_generate()
