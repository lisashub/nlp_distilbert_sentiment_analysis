
from transformers import TFAutoModel, AutoTokenizer, logging
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import random
import numpy as np


#Data Prep Function by Dr. Rohit Kate (COMPSCI723 - Fall 2021)
def readTextExamples(folder,cl,n) :
    """ Reads maximum n text files from folder and returns them as a list of
        list of text and its class label cl."""
    from glob import glob
    x = [] # list of text examples and class labels

    files = glob(folder+"/*.txt") # read all text files
    for file in files :
        infile = open(file,"r",encoding="utf-8")
        data = infile.read()
        infile.close()
        x.append([data,cl])
        if len(x)==n :
            break

    return x

#Data Prep Function by Dr. Rohit Kate (COMPSCI723 - Fall 2021)
def readPosNeg(pos_folder,neg_folder,n) :
    """ Reads maximum n positive and maximum n negative text examples
        from the respective folders. Randomizes them. Returns the list of
        text and an np.array of corresponding one-hot class labels [1,0] for pos
        and [0,1] for neg. """
    pos = readTextExamples(pos_folder,[1,0],n)
    neg = readTextExamples(neg_folder,[0,1],n)
    allEg = pos + neg
    random.shuffle(allEg)
    x = []
    y = []
    for eg in allEg :
        x.append(eg[0])
        y.append(eg[1])
    return x, np.array(y)

#Shared Model Parameters
maxlen = 512 # maximum number of tokens
maxTrEg = 1000 # maximum number of training examples
maxTeEg = 1000 # maximum number of test examples
token_ids = Input(shape=(maxlen,), dtype=tf.int32, name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32, name="attention_masks")
logging.set_verbosity_error() #supresses transformers model checkpoint warnings

#Training & Testing Data Prep
train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg) #Set ACL dataset train data
test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg) #Set ACL dataset test data
small_test_x, small_test_y = readPosNeg("small/test/pos", "small/test/neg",maxTeEg) #Set small dataset test data


#****************DistilBERT Base Uncased****************************


#Configure Tokenizer
dbu_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


#Tokenize Training and Test Datasets
tokenized_train = dbu_tokenizer(train_x, max_length=maxlen, truncation=True, padding=True, return_tensors="tf")
tokenized_test = dbu_tokenizer(test_x, max_length=maxlen, truncation=True, padding=True, return_tensors="tf")
small_tokenized_test = dbu_tokenizer(small_test_x, max_length=maxlen, truncation=True, padding=True, return_tensors="tf")

#Build Model
distilbert_base_uncased = TFAutoModel.from_pretrained("distilbert-base-uncased")
distilbert_base_uncased.trainable = False
dbu_output = distilbert_base_uncased(token_ids,attention_mask=attention_masks)
dbukerasoutput = Dense(2,activation="softmax")(dbu_output[0][:,0])
dbukerasmodel = Model(inputs=[token_ids,attention_masks],outputs=dbukerasoutput)

#Compile Model
dbukerasmodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Fit Model
print("\n")
print("Fitting DistilBERT Base Uncased model to ACL training data....")
dbukerasmodel.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],train_y, batch_size=1, epochs=3)


#Evaluate Model on ACL Test Data
print("\n")
print("Evaluating DistilBERT Base Uncased model on ACL test dataset....")
score = dbukerasmodel.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,batch_size = 1, verbose=1)
print("Accuracy on ACL test dataset:",score[1])


#Evaluate Model on Small Test Data
print("\n")
print("Evaluating DistilBERT Base Uncased model on small test dataset....")
small_score = dbukerasmodel.evaluate([small_tokenized_test["input_ids"],small_tokenized_test["attention_mask"]],small_test_y, batch_size = 1, verbose=1)
print("Accuracy on small test dataset:", small_score[1])


#Review Small Test Predictions
print("\n")
print("DistilBERT Base Uncased small dataset test example predictions....")
small_predictions = dbukerasmodel.predict([small_tokenized_test["input_ids"], small_tokenized_test["attention_mask"]])
i = 0
for elem in small_predictions:
    print("Prediction for test example " + str(i+1) + ": " + str(elem))
    print("Test example text:" + "\n" + small_test_x[i] + "\n")
    i = i + 1



#*********DistilBERT Base Uncased SQuAD**************

#Configure Tokenizer
dbs_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

#Tokenize Training and Test Datasets
tokenized_train = dbs_tokenizer(train_x, max_length=maxlen, truncation=True, padding=True, return_tensors="tf")
tokenized_test = dbs_tokenizer(test_x, max_length=maxlen, truncation=True, padding=True, return_tensors="tf")
small_tokenized_test = dbs_tokenizer(small_test_x, max_length=maxlen, truncation=True, padding=True, return_tensors="tf")

#Build Model
distilbert_base_squad = TFAutoModel.from_pretrained("distilbert-base-uncased-distilled-squad")
distilbert_base_squad.trainable = False
dbs_output = distilbert_base_squad(token_ids,attention_mask=attention_masks)
output = Dense(2,activation="softmax")(dbs_output[0][:,0])
dbskerasmodel = Model(inputs=[token_ids,attention_masks],outputs=output)

#Compile Model
dbskerasmodel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Fit Model to ACL training data
print("\n")
print("Fitting DistilBERT Base Uncased SQuAD model to ACL training data....")
dbskerasmodel.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],train_y, batch_size=1, epochs=3)

#Evaluate Model on ACL Test Dataset
print("\n")
print("Evaluating DistilBERT Base Uncased SQuAD model on ACL test dataset....")
score = dbskerasmodel.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y, batch_size = 1, verbose=1)
print("Accuracy on ACL test dataset:",score[1])

#Evaluate Model on Small Test Dataset
print("\n")
print("Evaluating DistilBERT Base Uncased SQuAD model on small test dataset....")
small_score = dbskerasmodel.evaluate([small_tokenized_test["input_ids"],small_tokenized_test["attention_mask"]],small_test_y, batch_size = 1, verbose=1)
print("Accuracy on small test dataset:", small_score[1])

#Review Small Test Dataset Predictions
print("\n")
print("DistilBERT Base Uncased SQuAD small dataset test example predictions....")
small_predictions = dbskerasmodel.predict([small_tokenized_test["input_ids"], small_tokenized_test["attention_mask"]], batch_size=1)
i = 0
for elem in small_predictions:
    print("Prediction for test example " + str(i+1) + ": " + str(elem))
    print("Test example text:" + "\n" + small_test_x[i] + "\n")
    i = i + 1


