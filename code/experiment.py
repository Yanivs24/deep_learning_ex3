#!/usr/bin/python

import numpy as np
import dynet as dy
import random


STUDENT={'name': 'Yaniv Sheena',
         'ID': '308446764'}

TRAIN_FILE = 'train_set'
TEST_FILE  = 'test_set'

def read_examples(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    return [l.strip().split() for l in lines]


class dynet_model:
    def __init__(self, indexed_vocab, indexed_labels, rnn_input=50, rnn_output=50, mlp_hid_dim=20):

        self.indexed_vocab = indexed_vocab
        self.indexed_labels = indexed_labels
        # reverse dict - from index to label
        self.i2l = {i: label for label, i in indexed_labels.iteritems()}

        self.vocab_size = len(indexed_vocab)
        self.out_dim = 2 # binary classifier

        # define the parameters
        self.model = dy.Model()

        # build LSTM
        self.rnn = dy.LSTMBuilder(1, rnn_input, rnn_output, self.model)

        # word embedding - E
        self.E = self.model.add_lookup_parameters((self.vocab_size,rnn_input))

        # first MLP layer params
        self.pW1 = self.model.add_parameters((mlp_hid_dim, rnn_output))
        self.pb1 = self.model.add_parameters(mlp_hid_dim)

        # hidden MLP layer params
        self.pW2 = self.model.add_parameters((self.out_dim, mlp_hid_dim))
        self.pb2 = self.model.add_parameters(self.out_dim)

       
    def predict_labels(self, seq):
        # Propagate over the RNN recursively
        s = self.rnn.initial_state()
        for char in seq:
            s = s.add_input(self.E[indexed_vocab[char]])            
        x = s.output()
        h = self.layer1(x)
        y = self.layer2(h)
        return dy.softmax(y)

    def layer1(self, x):
        W = dy.parameter(self.pW1)
        b = dy.parameter(self.pb1)
        return dy.tanh(W*x+b)

    def layer2(self, x):
        W = dy.parameter(self.pW2)
        b = dy.parameter(self.pb2)
        return W*x+b

    def do_loss(self, probs, label):
        label = self.indexed_labels[label]
        return -dy.log(dy.pick(probs,label))

    def classify(self, seq, label):
        dy.renew_cg()
        probs = self.predict_labels(seq)
        vals = probs.npvalue()
        return np.argmax(vals), -np.log(vals[label])

    def predict(self, seq):
        ''' classify without the loss '''
        prediction, _ = self.classify(seq, 0)
        return prediction

    def train_model(self, train_data, dev_data, learning_rate=1, max_iterations=20):
        #trainer = dy.SimpleSGDTrainer(self.model)
        trainer = dy.AdamTrainer(self.model)
        best_dev_loss = 1e3
        best_iter = 0
        print 'Start training the model..'
        for ITER in xrange(max_iterations):
            random.shuffle(train_data)
            closs = 0.0
            train_success = 0
            for seq, label in train_data:
                dy.renew_cg()
                probs = self.predict_labels(seq)
                loss = self.do_loss(probs,label)
                closs += loss.value()
                loss.backward()
                trainer.update(learning_rate)

                vals = probs.npvalue()
                if self.i2l[np.argmax(vals)] == label:
                    train_success += 1
                
            # check performance on dev set
            success_count = 0
            dev_closs = 0.0
            for seq, label in dev_data:
                real_label = indexed_labels[label]
                prediction, dev_loss = self.classify(seq, real_label)
                # accumulate loss
                dev_closs += dev_loss
                success_count += (prediction == real_label)

            avg_dev_loss = dev_closs/len(dev_data)
            dev_accuracy = float(success_count)/len(dev_data)

            print "Train accuracy: %s | Dev accuracy: %s | Dev avg loss: %s" % (float(train_success)/len(train_data), 
                dev_accuracy, avg_dev_loss)


        print 'Learning process is finished!'


if __name__ == '__main__':

    train_data = read_examples(TRAIN_FILE)
    test_data  = read_examples(TEST_FILE)

    print 'Got %s train examples and %s test examples' % (len(train_data), len(test_data))

    vocab = map(str,range(1, 10)) + ['a','b','c','d']
    #vocab = ['a', 'b']
    labels = ['pos', 'neg']

    indexed_vocab = {w: i for i,w in enumerate(vocab)}
    indexed_labels = {l: i for i,l in enumerate(labels)}

    # build a new dynet model 
    my_model = dynet_model(indexed_vocab, indexed_labels)

    # train the model
    my_model.train_model(train_data, test_data)