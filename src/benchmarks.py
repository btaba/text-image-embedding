"""
Run benchmarks on image and text vectors
"""
import click
import tabulate
import numpy as np
from utils import data_utils
from utils.data_utils import open_dataset
from utils.data_utils import stream_json, BASE_PATH
from sklearn.neighbors import NearestNeighbors

CLIP_RANKING = int(5e4)  # max number of neighbors to measure ranking benchmark


@click.group()
def cli():
    pass


def load_split_image_ids(split, dataset, encoding_name):
    """Given a split and dataset, return the split and its (indexes, image ids)"""
    # read in image-IDs for each row
    json_file = BASE_PATH / dataset / encoding_name /\
        '{}-encoded-captions-and-images.json'.format(split)
    image_ids = [image['id'] for image in stream_json(json_file)]

    idx2img = {i: image_id for i, image_id in enumerate(image_ids)}
    img2idx = {image_id: i for i, image_id in enumerate(image_ids)}
    return idx2img, img2idx, image_ids


def load_idx_to_captions(path, split):
    json_file = path /\
        '{}-encoded-captions-and-images.json'.format(split)
    idx2captions = {i: image['text'] for i, image in enumerate(stream_json(json_file))}
    img2captions = {image['id']: image['text'] for image in stream_json(json_file)}
    return idx2captions, img2captions


def recall_benchmarks(neighbors_model, C, top_k, ground_truth_ids, idx2img):
    nearest = neighbors_model.kneighbors(C, n_neighbors=top_k)[1]

    # Top_k predicted images for each sentence
    nearest = [ni[:top_k] for ni in nearest]
    nearest = [[idx2img[x] for x in ni] for ni in nearest]

    # Now compare predictions to ground truth
    comparable = list(zip(ground_truth_ids, nearest))

    recall = [1.0 if gt in ni else 0.0 for gt, ni in comparable]

    average_recall = np.mean(recall)

    return average_recall


def rank_benchmarks(neighbors_model, C, n_neighbors,
                    ground_truth_ids, idx2img, nearest_index_start=0):
    # avoid memory errors by clipping the ranking
    min_n_neighbors = min(n_neighbors, CLIP_RANKING)
    
    nearest = neighbors_model.kneighbors(C, n_neighbors=min_n_neighbors)[1]
    nearest = [[idx2img[x] for x in ni[nearest_index_start:]] for ni in nearest]

    comparable = list(zip(ground_truth_ids, nearest))

    rank = []
    for gt, ni in comparable:
        if gt in ni:
            rank.append(ni.index(gt) + 1)
        else:
            rank.append(n_neighbors)

    median_rank = np.median(rank)
    mean_rank = np.mean(rank)

    return median_rank, mean_rank


def visualize_image_annotations(encoding_name, dataset, split, neighbors_model, idx2img, C,
                                ground_truth_captions, n_neighbors=5, top_k=10):
    image_path = open_dataset(dataset)[ground_truth_captions[0]].parents[0]
    output_file = image_path / 'image_annotations.html'

    idx2captions, img2captions = load_idx_to_captions(
        BASE_PATH / dataset / encoding_name, split)

    nearest = neighbors_model.kneighbors(C, n_neighbors=n_neighbors)[1]
    nearest = [[(idx2img[x], idx2captions[x]) for x in ni] for ni in nearest]
    ground_truth = [(g, img2captions[g]) for g in ground_truth_captions]
    comparable = list(zip(ground_truth, nearest))

    def make_meta(c):
        gt = c[0]
        nearest = c[1]
        return "{} :::: {}".format(gt[1], nearest)

    html_metadata = [(str(c[0][0]), make_meta(c)) for c in comparable]
    print('Writing demo html to {}'.format(str(output_file)))
    data_utils.images_to_html(html_metadata[:top_k], str(output_file))


def visualize_image_search(encoding_name, dataset, split, neighbors_model, idx2img,
                           C, ground_truth_image_ids, n_neighbors=5, top_k=100):
    image_path = open_dataset(dataset)[ground_truth_image_ids[0]].parents[0]
    output_file = image_path / 'image_search.html'

    idx2captions, img2captions = load_idx_to_captions(
        BASE_PATH / dataset / encoding_name, split)

    nearest = neighbors_model.kneighbors(C, n_neighbors=n_neighbors)[1]
    nearest = [[(idx2img[x], img2captions[idx2img[x]]) for x in ni] for ni in nearest]
    ground_truth_captions = [imid + ' ' + idx2captions[i] for i, imid in enumerate(ground_truth_image_ids)]
    comparable = list(zip(ground_truth_captions, nearest))

    html_metadata = {}
    for i, c in enumerate(comparable):
        if i > top_k:
            break
        html_metadata[c[0]] = [(im[0], im) for im in c[1]]

    print('Writing demo html to {}'.format(str(output_file)))
    data_utils.images_to_html_grouped_by_key(html_metadata, str(output_file))


def nn_rank_word_vectors(dataset, encoding_name, split, Y_c):
    """
    Rank nearest neighbor word vector rankings for captions
    """
    # Get indexes to a split's image ids (ground truth)
    idx2img, img2idx, image_ids = load_split_image_ids(split, dataset, encoding_name)
    text_search = NearestNeighbors(10, algorithm='brute', metric='cosine')
    text_search.fit(Y_c)
    median, mean = rank_benchmarks(
        text_search, Y_c, Y_c.shape[0], image_ids, idx2img,
        nearest_index_start=1)
    print('median: {}, mean: {}'.format(median, mean))


def benchmark_func(dataset, encoding_name, split,
                   X_c, Y_c, distance, visualize=False):

    # Get indexes to a split's image ids (ground truth)
    idx2img, img2idx, image_ids = load_split_image_ids(split, dataset, encoding_name)

    # X_c is not unique, so we need to make it unique (image:text is 1:Many)
    ordered_image_idx_set = []
    for i in image_ids:
        if i not in ordered_image_idx_set:
            ordered_image_idx_set.append(i)
    ordered_unique_image_idxs = [img2idx[i] for i in ordered_image_idx_set]
    X_c_sliced = X_c[ordered_unique_image_idxs]
    sliced_image_ids = [idx2img[i] for i in ordered_unique_image_idxs]
    sliced_idx2img = {i: img for i, img in enumerate(sliced_image_ids)}

    # 10 for max recall in benchmark
    text_search = NearestNeighbors(10, algorithm='brute', metric=distance)
    text_search.fit(Y_c)

    # again, 10 for max recall in benchmark
    image_search = NearestNeighbors(10, algorithm='brute', metric=distance)
    # Need unique images, so not X_c directly sincse there are several captions per image
    image_search.fit(X_c_sliced)

    top_ks = [1, 5, 10]
    image_annotation_results, image_search_results = [], []
    for top_k in top_ks:
        # Image annotation Benchmarks
        recall = recall_benchmarks(text_search, X_c_sliced, top_k, sliced_image_ids, idx2img)
        image_annotation_results.append(recall)

        # Image search
        # for each caption, get top-k neighbors of images and get recall
        recall = recall_benchmarks(image_search, Y_c, top_k, image_ids, sliced_idx2img)
        image_search_results.append(recall)

    # Image annotation rank
    median_rank, mean_rank = rank_benchmarks(text_search, X_c_sliced, Y_c.shape[0],
                                             sliced_image_ids, idx2img)
    image_annotation_results.extend([median_rank, mean_rank])
    headers = ['recall@{}'.format(k) for k in top_ks] + ['median_rank', 'mean_rank']
    print('Image Annotation')
    imannotation = tabulate.tabulate([image_annotation_results], headers=headers)
    print(imannotation)
    
    if visualize:
        visualize_image_annotations(encoding_name, dataset, split,
                                    text_search, idx2img,
                                    X_c_sliced, sliced_image_ids)

    # Image search rank
    median_rank, mean_rank = rank_benchmarks(image_search, Y_c, X_c_sliced.shape[0],
                                             image_ids, sliced_idx2img)
    image_search_results.extend([median_rank, mean_rank])
    print('Image Search')
    imsearch = tabulate.tabulate([image_search_results], headers=headers)
    print(imsearch)

    if visualize:
        visualize_image_search(encoding_name, dataset, split,
                               image_search, idx2img,
                               Y_c, image_ids)

    return {
        'image_search': dict(zip(headers, image_search_results)),
        'image_annotation': dict(zip(headers, image_annotation_results))
    }


@click.command()
@click.argument('vectors_path')
@click.argument('dataset')
@click.argument('split')
@click.argument('encoding_name')
@click.option('--distance', default='cosine')
@click.option('--visualize', default=False)
def run_benchmark(vectors_path, dataset, split, encoding_name, distance, visualize):
    """
    Run Image Search and Immage Annotation benchmarks on a captioned image dataset.
    You must provide encoded embeddings for images and captions stored in the vectors_path.
    """
    vectors_path = BASE_PATH / dataset / vectors_path

    X_c = np.load(vectors_path / encoding_name / ('{}_X_c.npy'.format(split)))  # image
    Y_c = np.load(vectors_path / encoding_name / ('{}_Y_c.npy'.format(split)))  # text

    benchmark_func(dataset, encoding_name, split, X_c, Y_c, distance, visualize)


cli.add_command(run_benchmark)


if __name__ == '__main__':
    cli()