# NLP DistilBERT Sentiment Analysis

## Description
This project consists of two models: (1) DistilBERT base uncased and (2) DistilBERT base uncased  w/ an additional linear layer trained on SQuAD v.1.1. 
Each model is trained with 1000 examples from ACL IMDB dataset, then tested with 1000 ACL IMBD dataset examples followed by a small handcrafted dataset of 20 examples of 
IMDB reviews for the movie Life Aquatic. The resulting models output  prdictions and confidence on whether a given set of text reflects a positive or negative sentiment.

In terms of structure, each base model is provided an additional dense Keras softmax output layer. DistilBERT base models' parameters are fixed and not adjusted during 
training; only the dense output layers are fine-tuned. 

