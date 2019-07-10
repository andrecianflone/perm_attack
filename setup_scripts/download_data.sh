#!/bin/bash
# Prep dir
cd ..
mkdir -p _data && cd _data

# Download from gdrive
fileid="1Nbnp37FSDE-6e_gIJUvRcoAeiVqHci13"
filename="data.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Extract
echo "Extracting IMDB, this may take a few minutes"
tar -xzf data.tar.gz
mv .data/ ../
cd ..
rm -rf _data/
