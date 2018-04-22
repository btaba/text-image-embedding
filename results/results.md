
Results on the test splits after doing cross-validation on train/val sets.

# Flickr8k

To generate the benchmarks, we did the following:

```
> python cca_generate.py cca flickr8k word2vec_vgg19
> python benchmarks.py run_benchmark cca flickr8k test word2vec_vgg19 --distance cosine
```


## Image Search


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 19.1 | 45.3 | 60.4 | 7 | 27.1 |
| Our CCA - Mean word2vec_vgg19 | 14.9 | 39.7 | 54.2 | 9 | 34.7 |
| Our CCA - Mean word2vec_inceptionresnetv2 | 18.3 | 43.5 | 57.7 | 7 | 30.9 |
| Our CCA - Mean fasttext_vgg19 | 14.5 | 38.4 | 52.2 | 9 | 36.2 |
| Our CCA - Mean numberbatch_vgg19 | 15.9 | 40.8 | 55.2 | 8 | 32.7 |
| Our CCA - Mean numberbatch_inceptionresnetv2 | 18.4 | 44.6 | 58.2 | 7 | 30.1 |
|  |  |  |  |  |  |


## Image Annotation


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 22.6 | 48.8 | 61.2 | 6 | 28.8 |
| Our CCA - Mean word2vec_vgg19 | 17.8 | 41.9 | 55.5 | 8 | 39.9 |
| Our CCA - Mean word2vec_inceptionresnetv2 | 20.9 | 45.5 | 59.9 | 7 | 29.856 |
| Our CCA - Mean fasttext_vgg19 | 18.0 | 38.9 | 51.6 | 10 | 43.6 |
| Our CCA - Mean numberbatch_vgg19 | 20.1 | 43.9 | 57.2 | 7 | 40.0 |
| Our CCA - Mean numberbatch_inceptionresnetv2 | 22.1 | 46.5 | 61.0 | 7 | 28.4 |
|  |  |  |  |  |  |



# Flickr30k

```
> python cca_generate.py cca flickr30k_images word2vec_vgg19
> python benchmarks.py run_benchmark cca flickr30k_images test word2vec_vgg19 --distance cosine
```


## Image Search


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 20.5 | 46.3 | 59.3 | 6.8 | 32.4 |
| Our CCA - Mean word2vec_vgg19 | 16.5 | 39.3 | 51.4 | 10 | 42.8 |
| Our CCA - Mean word2vec_inceptionresnetv2 | 21.36 | 47.9 | 61.4 | 6 | 32.2 |
| Our CCA - Mean fasttext_vgg19 | 16.1 | 37.0 | 50.0 | 11 | 45.5 |
| Our CCA - Mean numberbatch_vgg19 | 18.0 | 40.5 | 52.7 | 9 | 40.1 |
| Our CCA - Mean numberbatch_inceptionresnetv2 | 22.7 | 50.2 | 62.6 | 5 | 30.4 |
|  |  |  |  |  |  |


## Image Annotation


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 24.8 | 52.5 | 64.3 | 5 | 27.3 |
| Our CCA - Mean word2vec_vgg19 | 21.0 | 43.8 | 56.7 | 7 | 42.5 |
| Our CCA - Mean word2vec_inceptionresnetv2 | 22.1 | 50.5 | 62.4 | 5 | 27.5 |
| Our CCA - Mean fasttext_vgg19 | 19.8 | 44.6 | 55.8 | 8 | 48.5 |
| Our CCA - Mean numberbatch_vgg19 | 23.8 | 47.7 | 59.1 | 6 | 38.0 |
| Our CCA - Mean numberbatch_inceptionresnetv2 | 24.4 | 52.9 | 65.5 | 5 | 25.1 |
|  |  |  |  |  |  |


# MS COCO 2014



# References

[1] B. Klein, G. Lev, G. Sadeh, and L. Wolf, [“Fisher vectors derived from hybrid gaussian-laplacian mixture models for image annotation,”](https://www.cs.tau.ac.il/~wolf/papers/Klein_Associating_Neural_Word_2015_CVPR_paper.pdf) CVPR, 2015.
