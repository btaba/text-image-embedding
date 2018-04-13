#! /bin/bash

CURDIR=`pwd`
DATA_PATH=$HOME/data-text-image-embeddings/
cd $DATA_PATH

# download data from http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/
tar -xvf flickr30k.tar

mv results_20130124.token flickr30k_images
cd flickr30k_images
mkdir images
find . -name '*.jpg' -type f -maxdepth 1 -exec mv {} images/ \;

cd $CURDIR