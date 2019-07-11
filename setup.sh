#!/bin/bash

# This dir
root_dir=$(pwd)

# Make sure shell scripts are executable
find $rood_dir -type f -name "*.sh" -exec chmod 744 {} \;

echo "Install advertorch"
git clone https://github.com/BorealisAI/advertorch
cd advertorch
python setup.py install
cd ..
rm -rf advertorch

echo "******************"
echo "Installing python packages"
echo "******************"
pip install -r requirements.txt

echo "******************"
echo "Downloading glove"
echo "******************"
bash $root_dir/setup_scripts/download_glove.sh $root_dir/.vector_cache

echo "******************"
echo "Downloading data"
echo "******************"
bash $root_dir/setup_scripts/download_data.sh $root_dir/.data

echo "******************"
echo "Downloading pretrained LSTM"
echo "******************"
bash $root_dir/setup_scripts/download_pretrained.sh $root_dir/saved_models


