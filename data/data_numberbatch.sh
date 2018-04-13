#! /bin/bash

CURDIR=`pwd`
DATA_PATH=$HOME/data-text-image-embeddings/word-embeddings
mkdir -p $DATA_PATH
cd $DATA_PATH

wget https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz
gunzip numberbatch-en-17.06.txt.gz 

cd $CURDIR