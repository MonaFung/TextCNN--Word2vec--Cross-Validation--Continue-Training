# CNN-Text-Classification--Word2vec--Cross-Validation--Continue-Training

This work is based on dennybritz’s model “cnn-text-classification-tf”. 
https://github.com/dennybritz/cnn-text-classification-tf

While using TextCNN model in practice, I made several improvements to make this model more flexible and practical.

1.	How to implement word2vec in Embedding layer ?
1)	First, produce word2vec file using genism. You can run like this:
import logging
import os
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence('your_train.txt')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)
model.wv.save_word2vec_format(' your_w2v_result.txt', binary=False)
2)	Then add this to your train.py:
tf.flags.DEFINE_string("word2vec", "your_w2v_result.txt", "Word2vec file with pre-trained embeddings (default: None)")

if FLAGS.word2vec:
    # initial matrix with random uniform
    initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
    # load vectors from the word2vec
    print("Load word2vec file {}\n".format(FLAGS.word2vec))
    with open(FLAGS.word2vec, "r") as f:
        for line in f:
            line_1 = line.split(' ')
            word=line_1[0]
            vec=line_1[1:]
            idx = vocab_processor.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.array(vec)
            else:
                pass
    sess.run(cnn.W.assign(initW))

2.	How to apply cross-validation while training ?
Please refer to train_cross_validation.py.

3.	How to continue training from a certain checkpoint ?
Sometimes we want to train new samples on previous trained model instead of train from scratch. You can simply use save and restore feature of tensorflow.
Please refer to continue_train.py.
4.	How to use probability as model’s output ?
Maybe you are not only want to know the prediction class of the sample, but also the confidence or say probability of the prediction. Furthermore, in the case of multi-class classification problem, you want to know the top N best predictions. You can try below:
1)	Modify text_cnn.py:
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
self.predictions = tf.argmax(self.scores, 1, name="predictions")
self.prob = tf.nn.softmax(self.scores,name="prob")
self.top_prob = tf.nn.top_k(self.prob,1,name="top_prob") #add this to output the probility of the most likely prediction

2)	Modify eval.py:
top_prob = graph.get_operation_by_name("output/top_prob").outputs[0]
