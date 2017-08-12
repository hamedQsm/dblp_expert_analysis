import nltk

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import enchant

import pandas as pd


class Helper:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.stopwords = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()

        self.english_dic = enchant.Dict("en_US")

    def preprocess_text(self, text):
        '''
        removes stop words, punctuations, and lemmatizes
        :param text: 
        :return: 
        '''
        stopwords_removed = ' '.join([i for i in str(text).lower().split() if i not in self.stopwords])
        punct_removed = ''.join(i for i in stopwords_removed if i not in self.punctuation)
        lemmatized = " ".join(self.lemmatize.lemmatize(i) for i in punct_removed.split())
        return lemmatized

    def is_english(self, text):
        '''
        Checks if the given text is English. If more than half of the words are English returns true.
        :param text: 
        :return: 
        '''
        is_en_arr = [int(self.english_dic.check(w)) for w in str(text).split()]
        is_en = sum(is_en_arr) / float(len(is_en_arr)) > .5
        return is_en

    def prepare_author_df(self, csv_file, min_num_pub=2, max_num_pub=100, row_num=None):
        '''
        reads the csv file, and for each author creates one doc captaining all papers titles concatenated 
        :param csv_file: csv file with each raw containnig one author and one title. 
               Multiple row may refer to one publication.
        :param min_num_pub: lower bound for number of publication to consider an author
        :param max_num_pub: upper bound for number of publication to consider an author
        :param row_num: total rows to consider from the read csv file
        :return: 
        '''
        dblp_df = pd.read_csv(csv_file).head(row_num)
        if row_num:
            dblp_df = dblp_df.head(row_num)
        # remove documents which are not in English
        print('Filtering out non-English articles...')
        dblp_en_df = dblp_df[dblp_df['title'].apply(self.is_english)]

        # Aggregating on authors (concatenating titles to form one document)
        def aggregate_docs(group):
            titles = [str(t) for t in group['title']]
            return pd.Series({'doc': ' '.join(titles),
                              'num': len(group)
                              })

        print('Aggregating on authors...')
        author_df = dblp_en_df.groupby('author').apply(aggregate_docs).reset_index()

        # Filtering authors with "few" and "many" publication
        author_df = author_df[(author_df['num'] >= min_num_pub) & (author_df['num'] <= max_num_pub)]

        # Preprocessing the documents
        print('Preprocessing documents (removing stop words, punctuations, and lemmatizing)...')
        docs = author_df['doc'].apply(self.preprocess_text)
        doc_list = [t.split() for t in docs]
        print ('Total number of processed documents: ', len(doc_list))

        del author_df
        del dblp_df
        del dblp_en_df

        return doc_list


