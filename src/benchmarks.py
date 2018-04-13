"""
Run benchmarks on image and text vectors
"""
import click
import tabulate
import numpy as np
# from utils import data_utils
# from utils.data_utils import open_dataset
from utils.data_utils import stream_json, BASE_PATH
from sklearn.neighbors import NearestNeighbors

S3_PATH = 'https://s3.amazonaws.com/...../{}'


@click.group()
def cli():
    pass


def load_split_image_ids(split, dataset, encoding_name):
    """Given a split and dataset, return the split and its (indexes, image ids)"""
    # read in image-IDs for each row
    json_file = BASE_PATH / dataset / encoding_name /\
        '{}-encoded-captions-and-images.json'.format(split)
    image_ids = [image['id'] for image in stream_json(str(json_file))]

    idx2img = {i: image_id for i, image_id in enumerate(image_ids)}
    img2idx = {image_id: i for i, image_id in enumerate(image_ids)}
    return idx2img, img2idx, image_ids


def load_idx_to_captions(split, dataset, encoding_name):
    json_file = BASE_PATH / dataset / encoding_name /\
        '{}-encoded-captions-and-images.json'.format(split)
    idx2captions = {i: image['text'] for i, image in enumerate(stream_json(str(json_file)))}
    img2captions = {image['id']: image['text'] for image in stream_json(str(json_file))}
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


def rank_benchmarks(neighbors_model, C, n_neighbors, ground_truth_ids, idx2img):
    nearest = neighbors_model.kneighbors(C, n_neighbors=n_neighbors)[1]
    nearest = [[idx2img[x] for x in ni] for ni in nearest]

    comparable = list(zip(ground_truth_ids, nearest))

    rank = [ni.index(gt) + 1 for gt, ni in comparable]

    median_rank = np.median(rank)
    mean_rank = np.mean(rank)

    return median_rank, mean_rank


# def s3ify_path(dataset, f):
#     r = S3_PATH.format(dataset)
#     r = r + f.split(dataset)[-1]
#     return r
    

# def visualize_image_annotations(dataset, split, vectors_path, neighbors_model, idx2img, C,
#                                 ground_truth_captions, n_neighbors=5, top_k=10):
#     output_path = BASE_PATH / dataset / vectors_path
#     output_file = output_path / 'image_annotations.html'
#     idx2captions, img2captions = load_idx_to_captions(split, dataset)

#     nearest = neighbors_model.kneighbors(C, n_neighbors=n_neighbors)[1]
#     nearest = [[(idx2img[x], idx2captions[x]) for x in ni] for ni in nearest]
#     ground_truth = [(g, img2captions[g]) for g in ground_truth_captions]
#     comparable = list(zip(ground_truth, nearest))

#     def make_meta(c):
#         gt = c[0]
#         nearest = c[1]
#         return "{} :::: {}".format(gt[1], nearest)

#     paths = open_dataset(dataset)
#     html_metadata = [(s3ify_path(dataset, paths[c[0][0]]), make_meta(c)) for c in comparable]
#     print('Writing demo html to {}'.format(str(output_file)))
#     data_utils.images_to_html(html_metadata[:top_k], str(output_file))


# def visualize_image_search(dataset, split, vectors_path, neighbors_model, idx2img,
#                            C, ground_truth_image_ids, n_neighbors=5, top_k=100):
#     output_path = BASE_PATH / dataset / vectors_path
#     output_file = output_path / 'image_search.html'
#     idx2captions, img2captions = load_idx_to_captions(split, dataset)

#     nearest = neighbors_model.kneighbors(C, n_neighbors=n_neighbors)[1]
#     nearest = [[(idx2img[x], img2captions[idx2img[x]]) for x in ni] for ni in nearest]
#     ground_truth_captions = [imid + ' ' + idx2captions[i] for i, imid in enumerate(ground_truth_image_ids)]
#     comparable = list(zip(ground_truth_captions, nearest))

#     paths = open_dataset(dataset)

#     html_metadata = {}
#     for i, c in enumerate(comparable):
#         if i > top_k:
#             break
#         html_metadata[c[0]] = [(s3ify_path(dataset, paths[im[0]]), im) for im in c[1]]

#     print('Writing demo html to {}'.format(str(output_file)))
#     data_utils.images_to_html_grouped_by_key(html_metadata, str(output_file))


def benchmark_func(vectors_path, dataset, encoding_name, split, X_c, Y_c):

    # Get indexes to a split's image ids (ground truth)
    idx2img, img2idx, image_ids = load_split_image_ids(split, dataset)

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
    text_search = NearestNeighbors(10, algorithm='brute', metric='euclidean')
    text_search.fit(Y_c)

    # again, 10 for max recall in benchmark
    image_search = NearestNeighbors(10, algorithm='brute', metric='euclidean')
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
    # if vectors_path is not None:
    #     visualize_image_annotations(dataset, split, vectors_path, text_search, idx2img,
    #                                 X_c_sliced, sliced_image_ids)

    # Image search rank
    median_rank, mean_rank = rank_benchmarks(image_search, Y_c, X_c_sliced.shape[0],
                                             image_ids, sliced_idx2img)
    image_search_results.extend([median_rank, mean_rank])
    print('Image Search')
    imsearch = tabulate.tabulate([image_search_results], headers=headers)
    print(imsearch)
    # if vectors_path is not None:
    #     visualize_image_search(dataset, split, vectors_path, image_search, sliced_idx2img,
    #                            Y_c, image_ids)

    results = {
        'image_search': dict(zip(headers, image_search_results)),
        'image_annotation': dict(zip(headers, image_annotation_results))
    }

    return results


@click.command()
@click.argument('vectors_path')
@click.argument('dataset')
@click.argument('split')
def run_benchmark(vectors_path, dataset, split):
    vectors_path = BASE_PATH / dataset / vectors_path

    X_c = np.load(vectors_path / ('{}_X_c.npy'.format(split)))  # image
    Y_c = np.load(vectors_path / ('{}_Y_c.npy'.format(split)))  # text

    benchmark_func(vectors_path, dataset, split, X_c, Y_c)


cli.add_command(run_benchmark)


if __name__ == '__main__':
    cli()