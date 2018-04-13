import os
import json
import click
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_NAME = os.environ.get('DATA_PATH', 'data-text-image-embeddings')
BASE_PATH = Path.home() / DATA_NAME / 'word-embeddings'
DEFAULT_NUMBERBATCH_FILE = 'numberbatch-en-17.06.txt'
DEFAULT_NUMBERBATCH_OUTPUT_FILE = 'numberbatch-en-17.06.json.txt'


@click.group()
def cli():
    pass


def parse_rows(input_path, output_path):

    with input_path.open('r') as f, output_path.open('wt') as fout:
        for idx, row in enumerate(f):
            row = row.split()

            if not idx % 50000:
                logger.info(idx)

            if idx == 0 or len(row) != 301:
                continue

            token = row[0]
            if len(token) <= 1:
                logger.debug('Skipping word {}'.format(row[0]))
                continue

            row_data = {row[0]: [float(r) for r in row[-300:]]}
            print(json.dumps(row_data), file=fout)


@click.command()
@click.argument('fasttext_filename')
def preprocess_fasttext(fasttext_filename):
    """
    Save .vec file to json key: value dictionary
    """
    path = BASE_PATH / fasttext_filename
    output_path = BASE_PATH / (fasttext_filename + '.json.txt')

    parse_rows(path, output_path)


@click.command()
def preprocess_numberbatch():
    path = BASE_PATH / DEFAULT_NUMBERBATCH_FILE
    output_path = BASE_PATH / DEFAULT_NUMBERBATCH_OUTPUT_FILE
    parse_rows(path, output_path)


cli.add_command(preprocess_fasttext)
cli.add_command(preprocess_numberbatch)

if __name__ == '__main__':
    cli()
