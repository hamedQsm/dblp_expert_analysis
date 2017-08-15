import numpy as np
import gensim
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pickle
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# TODO: try tf-idf as well
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec[next(iter(word2vec))])
        print ('DIM: ', self.dim)

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



class Word2vecModel:
    def initialize(self, embedding_path):
        self.w2v = {}
        # At this moment I am using GLOVE as an already trained embedding.
        with open(embedding_path, "rb") as lines:
            for line in lines:
                word = str(line.split()[0])
                # removing b' and ' from the beginning and ending of the word
                word = word[2: -1]
                vec = np.array(list(map(float, line.split()[1:])))
                self.w2v[word] = vec

    def transform(self, doc_list):
        print ('Obtaining the mean embeddings...')
        X_mean_embeded = MeanEmbeddingVectorizer(self.w2v).transform(doc_list)

        return X_mean_embeded



