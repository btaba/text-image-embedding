#! /bin/bash

CURDIR=`pwd`
DATA_PATH=$HOME/data-text-image-embeddings/word-embeddings
mkdir -p $DATA_PATH
cd $DATA_PATH

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
rm wiki-news-300d-1M-subword.vec.zip

cd $CURDIR