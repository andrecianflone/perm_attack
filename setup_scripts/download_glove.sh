cd ..
mkdir -p .vector_cache && cd .vector_cache
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
rm glove.6B.100d.txt
rm glove.6B.200d.txt
rm glove.6B.50d.txt
