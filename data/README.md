# Data

* Flickr30k, Plummer 2017
    - Got the data manually from: http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/
    - Plummer paper https://arxiv.org/pdf/1505.04870.pdf
* `data_flickr8k.sh`
    - Flickr8k, Hodosh 2013, http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html
    - http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html
* `data_coco.sh` - MS COCC 2014


# Word Embeddings

* Download word2vec embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). Place them in ~/data-text-image-embeddings/word-embeddings
* Download fasttext with `data_fasttext.sh` and preprocess the file with `preprocess_word_embeddings.py`
