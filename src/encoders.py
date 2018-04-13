"""
The goal of this script is to encode images and sentences
"""

import click
import json
import numpy as np

from utils.data_utils import BASE_PATH, open_dataset
from utils.encoding import fasttext_vectorizer, numberbatch_vectorizer, word2vec_vectorizer
from utils.encoding import get_vgg19

paths = open_dataset()


@click.command()
@click.argument('dataset')
@click.argument('encoding-name')
def sentence_and_image_representations(dataset, encoding_name):
    """
    Encode both image and sentence representations into same file
    since the captions are Many-to-One for each image
    """

    # vectorizer = numberbatch_vectorizer()
    # vectorizer = fasttext_average_vectorizer(stopwords_set=stopwords_set)
    vectorizer = word2vec_vectorizer()
    stream_encoder, imread = get_vgg19()

    with (BASE_PATH / dataset / 'captions.json').open() as fin:
        captions = json.load(fin)

    dataset_path = BASE_PATH / dataset
    with (dataset_path / 'splits.json').open('r') as fin:
        images = json.load(fin)

    encoding_path = BASE_PATH / dataset / encoding_name
    encoding_path.mkdir(exist_ok=True)

    count = 0
    for split in images:
        split_images = [i for i in images[split]]
        split_images_path = [paths[i] for i in images[split]]

        split_images_stream = (imread(i) for i in split_images_path)
        split_images_stream = stream_encoder(split_images_stream)
        caption_stream = [list(captions[imid].values()) for imid in split_images]

        encoded_filename = encoding_path / '{}-encoded-captions-and-images.json'.format(split)

        with encoded_filename.open('w') as fout:
            for image_id, capts, image_vec in zip(split_images, caption_stream, split_images_stream):
                print(count)
                count += 1
                for c in capts:
                    vec = vectorizer.transform([c]).tolist()[0]
                    
                    if not np.any(vec):
                        continue

                    # vec = (vec / np.sum(np.square(vec))).tolist()
                    print(json.dumps(
                          {'id': str(image_id),
                           'text': c,
                           'x_text': vec,
                           'x_image': image_vec.tolist()}), file=fout)


@click.group()
def cli():
    pass


cli.add_command(sentence_and_image_representations)


if __name__ == '__main__':
    cli()
