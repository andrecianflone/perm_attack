#!/bin/bash
# Prep dir

# Clean dir
mkdir -p $1 && cd $1
rm -rf $1/imdb

# Download from gdrive
mkdir -p temp && cd temp

fileid="1Nbnp37FSDE-6e_gIJUvRcoAeiVqHci13"
filename="data.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Extract
echo "Extracting IMDB, this may take a few minutes"
tar -xzf data.tar.gz
mv $1/temp/.data/imdb/ $1/
cd $1
rm -rf temp
rm $1/imdb/*gz
