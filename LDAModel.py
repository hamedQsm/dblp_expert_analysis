import logging
import gensim
from gensim import corpora
from time import time
import os
import pyLDAvis.gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# noinspection PyAttributeOutsideInit
class LDAModel:
    def __init__(self):
        self.dictionary = None
        self.corpus = None
        self.ldamodel = None

    def initialize(self, doc_list):
        '''
        Initialized the LDA model (creating the dictionary and corpus)
        :param doc_list: list of of documents.
        :return: 
        '''
        # Creating and saving the term dictionary of our courpus, where every unique term is assigned an index.
        self.dictionary = corpora.Dictionary(doc_list)
        self.dictionary.save('dictionary.dict')

        # Creating and saving the corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        corpora.MmCorpus.serialize('corpus.mm', self.corpus)

    def train(self, num_topics, doc_num_to_train_with=None):
        '''
        train the LDA model. if the model is not already initialized it tries to read the dictionary and corpus from disk
        :param num_topics: how many topic to include in the LDA model
        :param doc_num_to_train_with: How many of the documents to include for the training. 
        :return: 
        '''
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                            filename='running.log', filemode='w')

        # if not initialized already try to load from disk
        if not self.dictionary:
            if os.path.exists('dictionary.dict'):
                self.dictionary = gensim.corpora.Dictionary.load('dictionary.dict')
            else:
                print('Can not find the dictionary. Initialize the LDA.')
        if not self.corpus:
            if os.path.exists('corpus.mm'):
                self.corpus = gensim.corpora.MmCorpus('corpus.mm')
            else:
                print('Can not find the corpus. Initialize the LDA.')

        # if specified we train only on portion of corpus
        if doc_num_to_train_with:
            self.corpus = gensim.utils.ClippedCorpus(self.corpus, doc_num_to_train_with)

        start = time()
        print('Starting to train LDA model...')
        Lda = gensim.models.ldamodel.LdaModel
        # Running and Trainign LDA model on the document term matrix.
        self.ldamodel = Lda(self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=3, chunksize=100000)
        print('Training finished in: {:.2f}s'.format(time() - start))

        self.ldamodel.save('lda-model')

    def visualize(self, load_from_file=True):
        '''
        visualizes the topics with using pyLDAvis 
        :param load_from_file: 
        :return: 
        '''
        if load_from_file:
            self.dictionary = gensim.corpora.Dictionary.load('dictionary.dict')
            self.corpus = gensim.corpora.MmCorpus('corpus.mm')
            self.ldamodel = gensim.models.LdaModel.load('lda-model')

        lda_vis = pyLDAvis.gensim.prepare(self.ldamodel, self.corpus, self.dictionary)
        pyLDAvis.save_html(lda_vis, 'lda_vis.html')
        pyLDAvis.show(lda_vis)

    def generate_word_cloud(self, topic_id_list):
        '''
        Generates a word cloud for each of the given topics 
        :param topic_id_list: the list ot topics to generate the word cloud for.
        :return: 
        '''

        def terms_to_wordcounts(terms, multiplier=1000):
            return " ".join([" ".join(int(multiplier * i[1]) * [i[0]]) for i in terms])

        plt.figure(figsize=(15, 35))
        for i in topic_id_list:
            terms = [(self.dictionary.get(w_id[0]), w_id[1])
                     for w_id in self.ldamodel.get_topic_terms(i, topn=100)]
            wordcloud = WordCloud(background_color="black", height=400, width=800, max_font_size=100).generate(
                terms_to_wordcounts(terms))

            plt.subplot(5, 2, i + 1)
            # plt.figure(figsize=(10, 20))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title('Topic ' + str(i), fontsize=20)
            plt.tight_layout()

    def preprocess_new_doc(self, word_list):

        if not self.dictionary:
            self.dictionary = gensim.corpora.Dictionary.load('dictionary.dict')
        transfered_to_dic = self.dictionary.doc2bow(word_list)
        return transfered_to_dic

    def extract_topics(self, word_list):
        if not self.ldamodel:
            self.ldamodel = gensim.models.LdaModel.load('lda-model')
        topics = self.ldamodel[self.preprocess_new_doc(word_list)]
        return topics
