1) word embeddings - map words into vectors - word2vc and GloVe

2) Deep Neural Networks for NLP (Recurrent Neural Network)

3) recursive Neural Network(how do we make sense of sentences?) -> a sentence is more like a tree

reference - https://github.com/lazyprogrammer/machine_learning_examples
nlp_classes2

https://dumps.wikimedia.org
Download some of them 


convert from XML to TXT
https://github.com/yohasebe/wp2txt

sudo gem install wp2xt
wp2txt -i <filename>
--------------

wordcount/document
how many times this word appeared in the document / how many documents got this word in total . => tf-idf

shape of tabls
V = vocabulary size(# of total words)
D = vector Dimensionality

word embedding => just a vector that represents a word

### word analogies
king-queen . = prince-princess

France-Paris = Germany - Berlin

Japan-Japanese = China-chinese

walk-walking = swim - swimming

### How to find analogies
there are 4 words in every analogy
input 3 words and find 4th
eg-
king-man = ? - woman
