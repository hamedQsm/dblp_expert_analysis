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

    def prepare_author_df(self, csv_file, root_folder, min_num_pub=3, max_num_pub=100, data_portion=None):
        '''
        reads the csv file, and for each author creates one doc captaining all papers titles concatenated 
        :param csv_file: csv file with each raw containnig one author and one title. 
               Multiple row may refer to one publication.
        :param min_num_pub: lower bound for number of publication to consider an author
        :param max_num_pub: upper bound for number of publication to consider an author
        :param data_portion: total rows to consider from the read csv file
        :return: 
        '''
        dblp_df = pd.read_csv(csv_file)
        if data_portion:
            print ('using ', int(len(dblp_df)*data_portion), ' of data.')
            dblp_df = dblp_df.head(int(len(dblp_df)*data_portion))
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
        author_df['doc'] = author_df['doc'].apply(self.preprocess_text)

        if data_portion:
            csv_name = '{}/data/{}-authors_processed.csv'.format(root_folder, data_portion)
        else:
            csv_name = '{}/data/authors_processed.csv'.format(root_folder)
        author_df.to_csv(csv_name, index=False)


        del dblp_df
        del dblp_en_df

        return author_df


