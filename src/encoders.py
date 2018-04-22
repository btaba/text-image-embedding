"""
The goal of this script is to encode images and sentences
"""

import click
import json
import numpy as np

from utils.data_utils import BASE_PATH, open_dataset
from utils.encoding import fasttext_vectorizer, numberbatch_vectorizer, word2vec_vectorizer
from utils.encoding import get_vgg19, get_nasnetlarge, get_inceptionresnetv2
from utils.encoding import get_inceptionresnetv2_tagger

paths = open_dataset()


def load_captions(dataset):
    with (BASE_PATH / dataset / 'captions.json').open() as fin:
        captions = json.load(fin)
    return captions


def load_images(dataset):
    dataset_path = BASE_PATH / dataset
    with (dataset_path / 'splits.json').open('r') as fin:
        images = json.load(fin)
    return images


def get_text_vectorizer(text_encoder):
    if text_encoder == 'numberbatch':
        vectorizer = numberbatch_vectorizer()
    elif text_encoder == 'fasttext':
        vectorizer = fasttext_vectorizer()
    elif text_encoder == 'word2vec':
        vectorizer = word2vec_vectorizer()
    else:
        raise NotImplementedError('{} not recognized text_encoder'.format(text_encoder))

    return vectorizer


def get_image_encoder(image_encoder):
    if image_encoder == 'vgg19':
        stream_encoder, imread = get_vgg19()
    elif image_encoder == 'nasnetlarge':
        stream_encoder, imread = get_nasnetlarge()
    elif image_encoder == 'inceptionresnetv2':
        stream_encoder, imread = get_inceptionresnetv2()
    else:
        raise NotImplementedError('{} not recognized image_encoder'.format(image_encoder))
    return stream_encoder, imread


def get_image_stream(images, imread, stream_encoder, split):
    split_images_path = [paths[i] for i in images[split]]
    split_images_stream = (imread(i) for i in split_images_path)
    split_images_stream = stream_encoder(split_images_stream)
    return split_images_stream


@click.command()
@click.argument('dataset')
@click.argument('text-encoder', type=click.Choice(['numberbatch', 'fasttext', 'word2vec']))
@click.argument('image-encoder', type=click.Choice(['vgg19', 'nasnetlarge', 'inceptionresnetv2']))
def sentence_and_image_representations(dataset, text_encoder, image_encoder):
    """
    Encode both image and sentence representations into same file
    since the captions are Many-to-One for each image.
    """

    vectorizer = get_text_vectorizer(text_encoder)
    stream_encoder, imread = get_image_encoder(image_encoder)

    captions = load_captions(dataset)
    images = load_images(dataset)

    encoding_path = BASE_PATH / dataset / ('{}_{}'.format(text_encoder, image_encoder))
    encoding_path.mkdir(exist_ok=True)

    count = 0
    for split in images:
        split_images = [i for i in images[split]]
        split_images_stream = get_image_stream(images, imread, stream_encoder, split)
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

                    print(json.dumps(
                          {'id': str(image_id),
                           'text': c,
                           'x_text': vec,
                           'x_image': image_vec.tolist()}), file=fout)


@click.command()
@click.argument('dataset')
@click.argument('text-encoder', type=click.Choice(['numberbatch', 'fasttext', 'word2vec']))
@click.argument('image-encoder', type=click.Choice(['vgg19', 'nasnetlarge', 'inceptionresnetv2']))
def image_tagger_representations(dataset, text_encoder, image_encoder):
    """
    Simply tag the images, and use the vectorized tags as the image representation.

    In the wild, the easiest thing to do is to tag images using some tagging network
        (trained on something like ImageNet) and then directly match search queries
        to those tags (or captions if we had them).

    Here we try to tag images using a network trained on ImageNet, and then use
        those tags as the image representations.
    """

    vectorizer = get_text_vectorizer(text_encoder)

    if image_encoder == 'inceptionresnetv2':
        stream_encoder, imread = get_inceptionresnetv2_tagger()
    else:
        raise NotImplementedError('{} not recognized image_encoder'.format(image_encoder))

    captions = load_captions(dataset)
    images = load_images(dataset)

    encoding_path = BASE_PATH / dataset / ('{}_{}'.format(text_encoder, image_encoder))
    encoding_path.mkdir(exist_ok=True)

    count = 0
    for split in images:
        split_images = [i for i in images[split]]
        split_images_stream = get_image_stream(images, imread, stream_encoder, split)
        caption_stream = [list(captions[imid].values()) for imid in split_images]

        encoded_filename = encoding_path / '{}-tagencoded-captions-and-images.json'.format(split)

        with encoded_filename.open('w') as fout:
            for image_id, capts, image_tags in zip(split_images, caption_stream, split_images_stream):
                print(count)
                count += 1
                for c in capts:
                    vec = vectorizer.transform([c]).tolist()[0]
                    image_vec = vectorizer.transform(image_tags).mean(axis=0).tolist()

                    print(json.dumps(
                          {'id': str(image_id),
                           'text': c,
                           'x_text': vec,
                           'x_image': image_vec}), file=fout)


@click.group()
def cli():
    pass


cli.add_command(sentence_and_image_representations)
cli.add_command(image_tagger_representations)


if __name__ == '__main__':
    cli()

