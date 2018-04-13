#! /bin/bash

CURDIR=`pwd`
DATA_PATH=$HOME/data-text-image-embeddings/mscoco
mkdir -p $DATA_PATH
cd $DATA_PATH

if [ ! -f train2014.zip ]; then
    wget http://images.cocodataset.org/zips/train2014.zip
fi

if [ ! -f val2014.zip ]; then
    wget http://images.cocodataset.org/zips/val2014.zip
fi

if [ ! -f test2014.zip ]; then
    wget http://images.cocodataset.org/zips/test2014.zip
fi

if [ ! -f annotations_trainval2014.zip ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
fi

if [ ! -f image_info_test2014.zip ]; then
    wget http://images.cocodataset.org/annotations/image_info_test2014.zip
fi

echo "Unzip"
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
unzip annotations_trainval2014.zip
unzip image_info_test2014.zip 

rm *.zip

echo "Done"
cd $CURDIR
