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
  
    path.mkdir(exist_ok=True, parents=True)

    np.save(path / ('{}_X_c'.format(split)), X_c)
    np.save(path / ('{}_Y_c'.format(split)), Y_c)
    # joblib.dump(model, path / 'cca.model')


def get_metrics_for_cv(r):
    """
    Metrics used for cross-validation
    """
    median_rank = r['image_annotation']['median_rank'] + r['image_search']['median_rank']
    recall_at_1 = r['image_annotation']['recall@1'] + r['image_search']['recall@1']
    return median_rank, recall_at_1


@click.command()
@click.argument('dataset')
@click.argument('encoding_name')
def cca_cv_fit(dataset, encoding_name):
    """
    Run cross-validation on a dataset and encoding using CCA
    """
    from models.cca import CCA

    X, Y = load_X_Y(dataset, 'train', encoding_name)
    X_val, Y_val = load_X_Y(dataset, 'validation', encoding_name)

    param_grid = {
        'n_components': [130],
        'center': [True],
        'reg': [0.01],
        'regscaled': [True],
        'scale_by_eigs': [True],
        'norm': [False],
        'distance': ['cosine']
    }

    results = []
    for params in ParameterGrid(param_grid):
        print(params)

        m = CCA().fit(X, Y, params['n_components'], params['reg'],
                      params['center'], params['regscaled'])

        X_c, Y_c = m.predict(X, Y, norm=params['norm'],
                             scale_by_eigs=params['scale_by_eigs'])
        r = benchmark_func(
            dataset, encoding_name, 'train', X_c, Y_c,
            params['distance'])
        train_median_rank, train_recall_at_1 = get_metrics_for_cv(r)

        X_c, Y_c = m.predict(X_val, Y_val, norm=params['norm'],
                             scale_by_eigs=params['scale_by_eigs'])
        r = benchmark_func(
            dataset, encoding_name, 'validation', X_c, Y_c,
            params['distance'])
        val_median_rank, val_recall_at_1 = get_metrics_for_cv(r)

        r = {
            'val_rank': val_median_rank, 'val_recall': val_recall_at_1,
            'train_rank': train_median_rank, 'train_recall': train_recall_at_1,
        }
        print(r)
        results.append((params, r))

    # get best train/val params
    best_params = sorted(results, key=lambda x: x[1]['val_rank'] + x[1]['train_rank'])
    print(best_params[:3])
    params = best_params[0]
    print('best params', params)


@click.command()
@click.argument('dataset')
@click.argument('encoding_name')
def cca_predict(dataset, encoding_name):
    """
    Fit CCA on a train dataset with encoding_name and predict on train/test/val.
    Saves prediction in encoding_name path.
    """
    from models.cca import CCA

    X, Y = load_X_Y(dataset, 'train', encoding_name)

    params = {
        'n_components': 130,
        'center': True,
        'reg': 0.01,
        'regscaled': True,
        'scale_by_eigs': True,
        'norm': False,
        'distance': 'cosine'
    }

    m = CCA().fit(X, Y, params['n_components'], params['reg'],
                  params['center'], params['regscaled'])

    # save predictions for benchmark
    for split in ['train', 'validation', 'test']:
        print('Saving {} embeddings'.format(split))
        X_s, Y_s = load_X_Y(dataset, split, encoding_name)
        X_c, Y_c = m.predict(X_s, Y_s, norm=params['norm'],
                             scale_by_eigs=params['scale_by_eigs'])
        save(X_c, Y_c, split, m, dataset, encoding_name)



cli.add_command(cca_cv_fit)
cli.add_command(cca_predict)

if __name__ == '__main__':
    cli()
