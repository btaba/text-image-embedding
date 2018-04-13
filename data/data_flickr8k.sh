#! /bin/bash
# Only for non-commercial use

CURDIR=`pwd`
DATA_PATH=$HOME/data-text-image-embeddings/flickr8k
mkdir -p $DATA_PATH
cd $DATA_PATH


echo "flickr8k"
if [ ! -f Flickr8k_Dataset.zip ]; then
    wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
fi

if [ ! -f Flickr8k_text.zip ]; then
    wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip
fi

echo "Unzip"
unzip Flickr8k_text.zip -d Flickr8k_text
unzip Flickr8k_Dataset.zip
rm *.zip

cd $CURDIR
