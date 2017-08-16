import logging
import gensim
from gensim import corpora
from time import time
import os
import pyLDAvis.gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import numpy as np


# noinspection PyAttributeOutsideInit
class LDAModel:
    def __init__(self, num_topics, name, root_folder):
        self.dictionary = None
        self.corpus = None
        self.ldamodel = None
        self.num_topics = num_topics
        self.name = name
        if not os.path.exists('{}/objects'.format(root_folder)):
            os.mkdir('{}/objects'.format(root_folder))
        self.dic_path = '{}/objects/{}-dictionary.dict'.format(root_folder, name)
        self.corpus_path = '{}/objects/{}-corpus.mm'.format(root_folder, name)
        self.model_path = '{}/objects/{}-lda-model'.format(root_folder, name)

    def initialize(self, doc_list):
        '''
        Initialized the LDA model (creating the dictionary and corpus)
        :param doc_list: list of of documents.
        :return: 
        '''
        # Creating and saving the term dictionary of our courpus, where every unique term is assigned an index.
        self.dictionary = corpora.Dictionary(doc_list)
        self.dictionary.save(self.dic_path)

        # Creating and saving the corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        corpora.MmCorpus.serialize(self.corpus_path, self.corpus)

    def fit(self, doc_num_to_train_with=None):
        '''
        train the LDA model. if the model is not already initialized it tries to read the dictionary and corpus from disk
        :param num_topics: how many topic to include in the LDA model
        :param doc_num_to_train_with: How many of the documents to include for the training. 
        :return: 
        '''
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                            filename='running.log', filemode='w')

        # self.num_topics = num_topics

        # if not initialized already try to load from disk
        if not self.dictionary:
            if os.path.exists(self.dic_path):
                self.dictionary = gensim.corpora.Dictionary.load(self.dic_path)
            else:
                print('Can not find the dictionary. Initialize the LDA.')
        if not self.corpus:
            if os.path.exists(self.corpus_path):
                self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
            else:
                print('Can not find the corpus. Initialize the LDA.')

        # if specified we train only on portion of corpus
        if doc_num_to_train_with:
            self.corpus = gensim.utils.ClippedCorpus(self.corpus, doc_num_to_train_with)

        start = time()
        print('Starting to train LDA model...')
        Lda = gensim.models.ldamodel.LdaModel
        # Running and Trainign LDA model on the document term matrix.
        self.ldamodel = Lda(self.corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=10, chunksize=10000)
        print('Training finished in: {:.2f}s'.format(time() - start))

        self.ldamodel.save(self.model_path)

    def visualize(self, load_from_file=True):
        '''
        visualizes the topics with using pyLDAvis 
        :param load_from_file: 
        :return: 
        '''
        if load_from_file:
            self.dictionary = gensim.corpora.Dictionary.load(self.dic_path)
            self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
            self.ldamodel = gensim.models.LdaModel.load(self.model_path)

        lda_vis = pyLDAvis.gensim.prepare(self.ldamodel, self.corpus, self.dictionary)
        pyLDAvis.save_html(lda_vis, 'lda_vis.html')
        pyLDAvis.show(lda_vis)

    def generate_word_cloud(self, topic_id_list):
        '''
        Generates a word cloud for each of the given topics 
        :param topic_id_list: the list ot topics to generate the word cloud for.
        :return: 
        '''

        def terms_to_wordcounts(terms, multiplier=10000):
            return " ".join([" ".join(int(multiplier * i[1]) * [i[0]]) for i in terms])


        for i in topic_id_list:
            terms = [(self.dictionary.get(w_id[0]), w_id[1])
                     for w_id in self.ldamodel.get_topic_terms(i, topn=100)]
            wordcloud = WordCloud(background_color="black", height=400, width=800, max_font_size=100).generate(
                terms_to_wordcounts(terms))

            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title('Topic ' + str(i), fontsize=20)
            plt.tight_layout()
            plt.savefig('Topic_'+ str(i))

    def load(self,  num_topics=100):
        '''
        Loads the trained lda model
        :param num_topics: number of topics, for the already trained model.
        :return: 
        '''
        if os.path.exists(self.dic_path):
            self.dictionary = gensim.corpora.Dictionary.load(self.dic_path)
        else:
            print('Can not find the dictionary. Initialize the LDA.')
        if os.path.exists(self.corpus_path):
            self.corpus = gensim.corpora.MmCorpus(self.corpus_path)
        else:
            print('Can not find the corpus. Initialize the LDA.')
        self.ldamodel = gensim.models.LdaModel.load(self.model_path)
        self.num_topics = num_topics

    def preprocess_new_doc(self, word_list):
        '''
        preprocess the new document, which is a list of words, to the dictionary representation.
        :param word_list: document. should be a list of words. 
        :return: 
        '''
        if not self.dictionary:
            self.dictionary = gensim.corpora.Dictionary.load(self.dic_path)
        transfered_to_dic = self.dictionary.doc2bow(word_list)
        return transfered_to_dic

    def extract_topics(self, word_list):
        '''
        retursn the list of topics of the given document.
        :param word_list: document. should be a list of words.
        :return: topics, sorted by level of relevance.
        '''
        if not self.ldamodel:
            self.load()
        topics = self.ldamodel[self.preprocess_new_doc(word_list)]
        return topics

    def transform(self, doc_list):
        '''
        transforms the a list of word lists to lda space
        :param doc_list: 
        :return: a matrix with each row as representation of the given document
        '''
        # TODO: I am sure this function can be implemented more efficiently
        topic_mat = np.zeros((len(doc_list), self.num_topics))
        for i, doc in enumerate(doc_list):
            topic_prob_vec = self.extract_topics(doc)
            for (t_id, p) in topic_prob_vec:
                topic_mat[i, t_id] = p
            if i % 1000 == 0:
                print ('Transferred ', i, ' documents.')

        return topic_mat