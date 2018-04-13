# Data

* Flickr30k from [Plummer 2017](https://arxiv.org/pdf/1505.04870.pdf)
    - Get the data manually from [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/), then see `data_flickr30k.sh` for extraction
* [Flickr8k](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html), [Hodosh 2013](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
    - Run `data_flickr8k.sh`
* MS COCC 2014
    - Run `data_coco.sh`


# Word Embeddings

* word2vec
    - Download word2vec embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). Place them in `~/data-text-image-embeddings/word-embeddings`
* fasttext
    - run `data_fasttext.sh` and preprocess the file with `preprocess_word_embeddings.py`
* numberbatch
    - run `data_numberbatch.sh` and preprocess the file with `preprocess_word_embeddings.py`
