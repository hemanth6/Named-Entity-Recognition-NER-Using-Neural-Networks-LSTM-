# Named-Entity-Recognition-NER-Using-Neural-Networks-LSTM-


This assignment gives you hands-on experience on building deep learning models on named entity recognition (NER). We will use the CoNLL-2003 corpus to build a neural network for NER. The same as HW3, in the folder named data, there are three files: train, dev and test. In the files of train and dev, we provide you with the sentences with human-annotated NER tags. In the file of test, we provide only the raw sentences. The data format is that, each line contains three items separated by a white space symbol. The first item is the index of the word in the sentence. The second item is the word type and the third item is the corresponding NER tag. There will be a blank line at the end of one sentence. We also provide you with a file named glove.6B.100d.gz, which is the GloVe word embeddings.


Task 1: Simple Bidirectional LSTM model

The first task is to build a simple bidirectional LSTM model (see slides page 43 in lecture 12 for the network architecture) for NER.
Implementing the bidirectional LSTM network with PyTorch. The architecture of the network is:
Embedding → BLSTM → Linear → ELU → classifier
Train this simple BLSTM model with the training data on NER with SGD as the optimizer. Please tune other parameters that are not specified in the above table, such as batch size, learning rate and learning rate scheduling.


Task 2: Using GloVe word embeddings
The second task is to use the GloVe word embeddings to improve the BLSTM in Task 1. The way we use the GloVe word embeddings is straight forward: we initialize the embeddings in our neural network with the corresponding vectors in GloVe. Note that GloVe is case-insensitive, but our NER model should be case-sensitive because capitalization is an important information for NER. You are asked to find a way to deal with this conflict. What are the precision, recall and F1 score on the dev data? (hint: the reasonable F1 score on dev is 88%.
