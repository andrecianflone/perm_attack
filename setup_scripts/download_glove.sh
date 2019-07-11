#!/bin/bash

# Prep dir
# $1 is something like .vector_cache/
# Clean dir
rm -rf $1
mkdir -p $1 && cd $1

echo "Downloading Glove, trained on 6B tokens Wiki + Gigaword"
mkdir temp && cd temp
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.300d.txt ../
cd ../
rm -rf temp/
