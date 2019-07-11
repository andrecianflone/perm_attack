#!/bin/bash
# Prep dir
mkdir -p $1 && cd $1

# Download from gdrive
fileid="1e88MXZiXIkXbwLbjTtjFb3Q2yRiB7XBI"
filename="lstm_torchtext2.pt"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

rm cookie
