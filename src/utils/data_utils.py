"""
Data utils
"""
import os
import json
from pathlib import Path

DATA_NAME = os.environ.get('DATA_PATH', 'data-text-image-embeddings')
BASE_PATH = Path.home() / DATA_NAME


def open_dataset(dataset_name='', data_path=BASE_PATH):
    """
    Create a dictionary of filename: path for a given dataset
    """
    path = data_path / dataset_name
    return {p.name: p for p in path.glob('**/*') if p.is_file()}


def stream_json(path):

    with path.open('rt') as f:
        for line in f:
            yield json.loads(line)


def images_to_html(images, output_html):
    """
    :param images: list of (link, text) for each image
    :param output_html: name of output html
    """

    with open(output_html, 'wt') as fout:
        print('<body>', file=fout)
        print('\t<ul>', file=fout)

        for link, metadata in images:
            print('\t\t<li><img src="{}" width="500"/><p>{}</p></li>'.format(
                link, metadata), file=fout)

        print('\t</ul>', file=fout)
        print('</body>', file=fout)


def images_to_html_grouped_by_key(images, output_html):
    """
    :param images: dict of imagename: (link, metadata)
    :param output_html: name of output html
    """

    MAIN_BLOCK = '''
        <div style="display: inline-block; margin-bottom: 50px;">
            <img src="{}" style="max-height: 300px; max-width: 300px;"/>
            <div>{}</div>
        </div>
    '''

    with open(output_html, 'wt') as fout:
        print('<body>', file=fout)
        print('\t<ul>', file=fout)

        for key in images:
            print('<h1>{}</h1><br/><br/>'.format(key), file=fout)
            for link, metadata in images[key]:
                print(MAIN_BLOCK.format(link, metadata), file=fout)
        print('\t</ul>', file=fout)
        print('</body>', file=fout)
