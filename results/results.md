
Results on test splits.

# Flickr8k

```
> python cca_generate.py cca flickr8k word2vec_vgg19
> python benchmarks.py run_benchmark cca flickr8k test word2vec_vgg19 --distance cosine
```

## Image Search


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 19.1 | 45.3 | 60.4 | 7.0 | 27.1 |
| Our CCA - Mean word2vec_vgg19 | 14.9 | 39.7 | 54.2 | 9 | 34.7 |
|  |  |  |  |  |  |


## Image Annotation


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 22.6 | 48.8 | 61.2 | 6.0 | 28.8 |
| Our CCA - Mean word2vec_vgg19 | 17.8 | 41.9 | 55.5 | 8 | 39.9 |
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
|  |  |  |  |  |  |


## Image Annotation


|       | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|-------|-----|-----|------|-------------|-----------|
| CCA Mean Vec [1] | 24.8 | 52.5 | 64.3 | 5 | 27.3 |
| Our CCA - Mean word2vec_vgg19 | 21.0 | 43.8 | 56.7 | 7 | 42.5 |
|  |  |  |  |  |  |



# MS COCO 2014


# References

[1] B. Klein, G. Lev, G. Sadeh, and L. Wolf, [“Fisher vectors derived from hybrid gaussian-laplacian mixture models for image annotation,”](https://www.cs.tau.ac.il/~wolf/papers/Klein_Associating_Neural_Word_2015_CVPR_paper.pdf) CVPR, 2015.
