# CNN-Text-Classification--Word2vec--Cross-Validation--Continue-Training

This work is based on dennybritz’s model “[cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)”. 

While using TextCNN model in practice, I made several improvements to make this model more flexible and practical.

## 1.How to implement word2vec in Embedding layer ?
### First, produce word2vec file using genism. 
You can run like this:<br>
import logging<br>
import os<br>
from gensim.models import word2vec<br>
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)<br>
sentences = word2vec.LineSentence('your_train.txt')<br>
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)<br>
model.wv.save_word2vec_format(' your_w2v_result.txt', binary=False)<br>
### Then, modify your train.py. 
### Please refer to train.py.

## 2.How to apply cross-validation while training ?
### Please refer to train_cross_validation.py.

## 3.How to continue training from a certain checkpoint ?
Sometimes we want to train new samples on previous trained model instead of train from scratch. You can simply use save and restore feature of tensorflow.
### Please refer to continue_train.py.

## 4.How to use probability as model’s output ?
Maybe you are not only want to know the prediction class of the sample, but also the confidence or say probability of the prediction. Furthermore, in the case of multi-class classification problem, you want to know the top N best predictions. You can try below:
### Modify text_cnn.py.
### Please refer to text_cnn.py.

### Modify eval.py.
### Please refer to eval.py.
