"""
Generate joint embeddings using CCA.
"""

import click
import numpy as np
# from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from benchmarks import benchmark_func
from utils.data_utils import stream_json, BASE_PATH


@click.group()
def cli():
    pass


def load_X_Y(dataset, split, encoding_name):
    json_file = BASE_PATH / dataset / encoding_name /\
        ('%s-encoded-captions-and-images.json' % split)

    X = [i['x_image'] for i in stream_json(json_file)]
    Y = [i['x_text'] for i in stream_json(json_file)]
    return np.array(X), np.array(Y)


def save(X_c, Y_c, split, model, dataset, encoding_name):
    """Save the encoded components and the model."""
    path = BASE_PATH / dataset / 'cca' / encoding_name
  
    path.mkdir(exist_ok=True)

    np.save(path / ('{}_X_c'.format(split)), X_c)
    np.save(path / ('{}_Y_c'.format(split)), Y_c)
    # joblib.dump(model, path / 'cca.model')


@click.command()
@click.argument('dataset')
@click.argument('encoding_name')
def cca(dataset, encoding_name):
    """Specify a dataset and an algorithm to perform cross decomposition."""
    from models.cca import CCA

    X, Y = load_X_Y(dataset, 'train', encoding_name)
    X_val, Y_val = load_X_Y(dataset, 'validation', encoding_name)

    param_grid = {
        'n_components': [65],
        'center': [True],
        'reg': [1e-2],
        'regscaled': [True],
        'scale_by_eigs': [True],
        'norm': [False]
    }

    results = []
    for params in ParameterGrid(param_grid):
        print(params)
        m = CCA().fit(X, Y, params['n_components'], params['reg'], params['center'], params['regscaled'])
        X_c, Y_c = m.predict(X_val, Y_val, scale_by_eigs=params['scale_by_eigs'])
        results.append((params, benchmark_func(None, 'flickr8k', encoding_name, 'validation', X_c, Y_c)))

    # sorted(results, key=lambda x: x[1]['image_annotation']['median_rank'] + x[1]['image_search']['median_rank'])[0]

    for split in ['train', 'validation', 'test']:
        X, Y = load_X_Y(dataset, split, encoding_name)
        X_c, Y_c = m.predict(X, Y)
        save(X_c, Y_c, split, m, dataset, encoding_name)


cli.add_command(cca)

if __name__ == '__main__':
    cli()
