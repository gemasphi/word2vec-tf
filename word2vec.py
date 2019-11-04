import numpy as np
import collections
import pandas as pd
import random
import math
import tensorflow as tf
from nltk.metrics.spearman import *
import os
from tensorflow.contrib.tensorboard.plugins import projector
from itertools import compress
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
import nltk as nltk
import gensim as gs
from six import iteritems
from scipy import spatial
from gensim.models import Word2Vec as w2v
import re
from copy import deepcopy

def calculate_cos_sim(embedding1, embedding2):
    # Must subtract one to get the similarity instead of the distance
    return 1 - spatial.distance.cosine(embedding1, embedding2)

class Word2Vec(object):
    SKIPGRAM = 'SKIPGRAM'
    CBOW = 'CBOW'
    #To build the graph when instantiated
    def __init__(self, model_name, batch_size, vocabulary_size, embedding_size, num_skips, skip_window, num_sampled, arch = SKIPGRAM):
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.num_sampled = num_sampled
        self.cos_sim_treshold = 0.05
        self.data_index = 0
        self.dictionary = {}
        self.sense_dict = {}
        self.sense_embedding = {}
        self.model_name = model_name
        self.arch = arch
        self._get_valid_examples = self._generate_valid_examples

    
    def _init_graph(self):
        print("Init graph")    
        print(self.batch_size)    

        # Input data.
        with tf.name_scope('inputs'):
            if(self.arch == Word2Vec.CBOW):
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_skips])
            else:
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])

            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:00'):

            with tf.name_scope('embeddings'):
                # Main embedding parameters
                self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

                # Embedded input vectors
                if(self.arch == Word2Vec.CBOW):
                    embed = tf.zeros([self.batch_size, self.embedding_size])
                    for i in range(self.num_skips):
                        embed += tf.nn.embedding_lookup(self.embeddings, self.train_inputs[:, i])
                    self.embed = embed
                else:
                    self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
 

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocabulary_size, self.embedding_size],
                        stddev=1.0 / math.sqrt(self.embedding_size)
                    )
                )
            with tf.name_scope('biases'):
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Calculate the NCE loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=self.embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size
                )
            )

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
        self.normalized_embeddings = self.embeddings / norm
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        self.init = tf.global_variables_initializer()

        # Create a saver.
        self.saver = tf.train.Saver()
        
        print("Finish init graph")    

    
    # To launch the graph
    def train(self, filename, epochs = 5):
        self.dictionary, self.dict_fname = self._build_dataset(filename, self.vocabulary_size)
        self.valid_examples = self._get_valid_examples()
        self.graph = self._init_graph()

        print('Initialized')

        with tf.Session(graph=self.graph) as session:
            self.init.run()
            average_loss = 0
            dicttoken2id = self.dictionary.token2id
            for i in range(epochs):
                print ('Epoch: ' + str(i))
                step = 0

                for batch_inputs, batch_labels in self._generate_batch(self.batch_size, dicttoken2id, self.num_skips, self.skip_window):
                    if(batch_inputs.size == 0):
                        break

                    step += 1
                    feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels }

                    # Actually run the tensorflow, get the output of the neural network
                    _, loss_val = session.run(
                        [self.optimizer, self.loss],
                        feed_dict=feed_dict
                    )
                    average_loss += loss_val

                    if step % 20000 == 0:
                        if step > 0:
                            average_loss /= 20000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step ', step, ': ', average_loss)
                        average_loss = 0

            embeddings_name = 'embed_{}_e_{}.npy'.format(self.model_name,self.embedding_size)
            self.final_embeddings = self.normalized_embeddings.eval()
            np.save(embeddings_name, self.final_embeddings)
            #self.final_embeddings = self.normalized_embeddings.eval()
            self.sim = self.similarity.eval()
            # Save the model for checkpoints.
            self.saver.save(session, '{}_e_{}.ckpt'.format(self.model_name,self.embedding_size))
        
        return embeddings_name, self.dict_fname

    def _build_dataset(self, filename, size):
        dict_fname = '{}_{}_dic'.format(filename,size)

        if (not os.path.isfile(dict_fname)):
            dictionary = gs.corpora.Dictionary(line.lower().split() for line in open(filename))
            stop_words = set(sw.words('english'))
            stop_ids = [dictionary.token2id[stopword] for stopword in stop_words
                        if stopword in dictionary.token2id]
            dictionary.filter_tokens(stop_ids)
            dictionary.filter_extremes(no_below = 0, no_above = 1, keep_n = size - 1)
            dictionary.compactify()
            dictionary.patch_with_special_tokens({'UNK': 0})
            dictionary.save_as_text(dict_fname)
        else:
            dictionary = gs.corpora.Dictionary.load_from_text(dict_fname)

        
        return dictionary, dict_fname

    def _get_pair(self, dictionary, num_skips, skip_window):
        with open('../data/text8') as f:
            for line in f:
                words = line.lower().split()
                size = len(words)
                data = []
                for word in words:
                    if word in dictionary:
                        data.append(dictionary[word])
                    else:
                        data.append(0) 
                
                if size > 2:
                    for i in range(size):
                        l = max(0, i - skip_window)
                        h = min(size - 1, i + skip_window)
                        context_words = [w for w in range(l, h + 1) if w != i] # context words

                        if(len(context_words) < num_skips):
                            words_to_use = random.sample(context_words, len(context_words)) # target word
                        else:
                            words_to_use = random.sample(context_words, num_skips) # target word
                            
                        for context_word in words_to_use:
                            yield data[i], data[context_word]

    def _generate_valid_examples(self):
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_size = 100

        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        
        return valid_examples
    
    def _generate_batch(self, batch_size, dictionary, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        i = 0
        for target,label in self._get_pair(dictionary, num_skips, skip_window):
            batch[i] = target
            labels[i,0] = label
            i += 1
            if i == batch_size:
                i = 0
                yield batch,labels

        yield np.array([]), np.array([])

    def calculate_spearmans_word_sim(self, embedding_file):
        embedding = dict(enumerate(np.load(embedding_file))) 

        wordsim = pd.read_csv('../data/combined.csv', header= None, skiprows = [0]) 
        wordsim[2] = wordsim[2]/5 - 1
        ws = []
        wv = []
        dicte = self.dictionary.token2id
        for i, row in wordsim.iterrows():
            w1,w2,score = row
            if (w1 in dicte) and (w2 in dicte):
                cos_sim = calculate_cos_sim(embedding[dicte[w1]], embedding[dicte[w2]])
                ws.append(((w1,w2),cos_sim))            
                wv.append(((w1,w2),score))
        
        ws = sorted(ws, key=lambda sim: sim[1])
        wv = sorted(wv, key=lambda sim: sim[1])
        
        spearman = spearman_correlation(list(ranks_from_scores(ws)), list(ranks_from_scores(wv)))

        return spearman


    def _generate_batch_cbow(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_ski36ps <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span) 
        # collect the first window of words
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        # move the sliding window  
        for i in range(batch_size):
            mask = [1] * span
            mask[skip_window] = 0 
            batch[i, :] = list(compress(buffer, mask)) # all surrounding words
            labels[i, 0] = buffer[skip_window] # the word at the center 
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        if(self.data_index == 0):
            print(batch)
            print(labels)
        return batch, labels

    def _norm_word(self, word):
        if re.compile(r'\d+.*').search(word.lower()):
            return '---num---'
        elif re.sub(r'\W+', '', word) == '':
            return '---punc---'
        else:
            return word.lower()

    def _read_lexicon(self, filename):
        lexicon = {}
        for line in open(filename, 'r'):
            words = line.lower().strip().split()
            lexicon[self._norm_word(words[0])] = [self._norm_word(word) for word in words[1:]]
        return lexicon

    def retrofit(self, embedding_file, lexicon_filename, dict_fname, num_iters = 10):
        self.dictionary = gs.corpora.Dictionary.load_from_text(dict_fname)

        word_vecs = {}
        for key, value in enumerate(np.load(embedding_file)): 
                word_vecs[self.dictionary[key]] = value
        
        lexicon = self._read_lexicon(lexicon_filename)
        new_word_vecs = deepcopy(word_vecs)
        wv_vocab = set(new_word_vecs.keys())
        loop_vocab = wv_vocab.intersection(set(lexicon.keys()))

        num_neighbours = 0
        for it in range(num_iters):
            for word in loop_vocab:
                word_neighbours = set(lexicon[word]).intersection(wv_vocab)
                num_neighbours = len(word_neighbours)
            if num_neighbours == 0:
                continue
            newVec = num_neighbours * word_vecs[word]
            for pp_word in word_neighbours:
                newVec += new_word_vecs[pp_word]
            new_word_vecs[word] = newVec/(2*num_neighbours)

        res = []
        for _, value in new_word_vecs.items():
            res.append(value)

        filename = 'improved_{}'.format(embedding_file)
        np.save(filename, res)
        return filename

    def compare_embeddings(self, emb1, emb2):
        emb1 = np.load(emb1) 
        emb2 = np.load(emb2) 

        cos_sim = []

        for i, value in enumerate(emb1):
            cos_sim.append(calculate_cos_sim(value, emb2[i]))

        return cos_sim