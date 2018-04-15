"""
Pre-process data
"""
import os
import json
import random
from collections import defaultdict
from utils import data_utils

paths = data_utils.open_dataset()
BASE_PATH = data_utils.BASE_PATH


def read_flickr_captions_file(captions_file):
    """
    Read a Flickr style captions file
    """
    captions = defaultdict(dict)
    with captions_file.open('rt') as f:
        for line in f.readlines():
            idx, caption = line.split('\t')
            image_id, caption_idx = idx.split('#')
            captions[image_id][caption_idx] = caption.strip()
    return captions


def shuffle_image_splits(all_images, num_images):
    random.seed(42)
    random.shuffle(all_images)
    splits = {
        'validation': all_images[:num_images],
        'test': all_images[num_images:(2 * num_images)],
        'train': all_images[(2 * num_images):]
    }
    return splits


def flickr8k():
    print('Processing flickr8k')

    # Captions
    captions_file = paths['Flickr8k.token.txt']
    captions = read_flickr_captions_file(captions_file)
    with (BASE_PATH / 'flickr8k' / 'captions.json').open('w') as f:
        json.dump(captions, f)

    # Splits
    splits = defaultdict(list)
    path_dict = {
        'train': paths['Flickr_8k.trainImages.txt'],
        'test': paths['Flickr_8k.testImages.txt'],
        'validation': paths['Flickr_8k.devImages.txt']
    }
    for split in path_dict:
        with path_dict[split].open('r') as f:
            for line in f.readlines():
                splits[split].append(line.strip())

    with (BASE_PATH / 'flickr8k' / 'splits.json').open('w') as f:
        json.dump(splits, f)

    print('Done with flickr8k')


def flickr30k():
    print('Processing flickr30k')
    # Captions
    captions_file = paths['results_20130124.token']
    captions = read_flickr_captions_file(captions_file)
    with (BASE_PATH / 'flickr30k_images' / 'captions.json').open('w') as f:
        json.dump(captions, f)

    # Splits - 1k for val and test
    # they all seem to eventually reference Hodosh https://www.jair.org/media/3994/live-3994-7274-jair.pdf
    # which doesn't mention the flickr 30k dataset at all
    # I'm just going to assume they do random splits
    all_images = os.listdir(str(BASE_PATH / 'flickr30k_images' / 'images'))
    splits = shuffle_image_splits(all_images, num_images=1000)
    with (BASE_PATH / 'flickr30k_images' / 'splits.json').open('w') as f:
        json.dump(splits, f)

    print('Done with flickr30k')


def mscoco():
    print('Processing MSCOC')

    # Captions
    all_images = []
    captions = defaultdict(dict)
    for split in ['val', 'train']:

        captions_file = paths['captions_{}2014.json'.format(split)]
        with captions_file.open('r') as f:
            split_captions = json.load(f)

        for c in split_captions['annotations']:
            image_id = 'COCO_{}2014_{:012d}.jpg'.format(split, c['image_id'])
            all_images.append(image_id)
            caption = c['caption']
            idx = c['id']
            captions[image_id][idx] = caption

    with (BASE_PATH / 'mscoco' / 'captions.json').open('w') as f:
        json.dump(captions, f)

    # Splits 5k for val and test
    # based on https://github.com/karpathy/neuraltalk2/blob/master/coco/coco_preprocess.ipynb
    # he concatenates the val + train images
    # then he does this https://github.com/karpathy/neuraltalk2/blob/master/prepro.py#L97
    # takes first 5k images as val, then 5k test, then train
    # first 5k is val, next 5k is test, then train
    splits = defaultdict(list)
    for idx, i in enumerate(all_images):
        if idx < 5000:
            split = 'validation'
        elif 10000 > idx >= 5000:
            split = 'test'
        else:
            split = 'train'
        splits[split].append(i)

    with (BASE_PATH / 'mscoco' / 'splits.json').open('w') as f:
        json.dump(splits, f)

    # more sensible splits
    splits = shuffle_image_splits(all_images, num_images=5000)
    with (BASE_PATH / 'mscoco' / 'more_sensible_splits.json').open('w') as f:
        json.dump(splits, f)

    print('Done with MSCOC')


if __name__ == '__main__':
    flickr8k()
    flickr30k()
    mscoco()
