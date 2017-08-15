import os
import pickle

import pandas as pd
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from wordcloud import WordCloud

from Helper import Helper
from models.LDAModel import LDAModel
from models.w2vecModel import Word2vecModel
import matplotlib.pyplot as plt


def jansen_shanon_div(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


test_authors = ['Hamed Ghasemieh'] #, 'Anne Remke', 'Boudewijn R. Haverkort']


def assess_model(model, model_name, distance_metric, data_portion=.7, load_pickles=True,
                 root_folder='.'):
    '''
    Assess the given model by returning the list of authors 
    :param root_folder: A folder containing both data, and object folders
    :param model_name: 
    :param load_pickles: 
    :param data_portion: 
    :param distance_metric: 
    :param model: 
    :return: 
    '''
    helper = Helper()
    # author_df = helper.prepare_author_df('{}/data/dblp_all.csv'.format(root_folder), root_folder,
    #                                      data_portion=data_portion, max_num_pub=1000)
    author_df = pd.read_csv('{}/data/{}-authors_processed.csv'.format(root_folder, data_portion))

    doc_list = [t.split() for t in author_df['doc']]
    print('Total number of processed documents: ', len(doc_list))

    if not load_pickles:
        print('Transforming to model space...')
        X = model.transform(doc_list)
        pickle.dump(X, open('{}/objects/{}-transferred_docs.pickle'.format(root_folder, model_name), "wb"))

        print('Training the nearest neighbor...')
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree', metric=distance_metric).fit(X)
        pickle.dump(nbrs, open('{}/objects/{}-nbrs-balltree.pickle'.format(root_folder, model_name), "wb"))
    else:
        # Loading the pickle in case we have already transformed the data.
        X = pickle.load(open('{}/objects/{}-transferred_docs.pickle'.format(root_folder, model_name), "rb"))
        # Loading the pickel in case we have already trained the model.
        nbrs = pickle.load(open('{}/objects/{}-nbrs-balltree.pickle'.format(root_folder, model_name), "rb"))

    # starting to query KNN
    query_authors = author_df[author_df['author'].isin(test_authors)]
    doc_list = [(helper.preprocess_text(p).split()) for p in query_authors['doc']]

    q = model.transform(doc_list)
    distances, indices = nbrs.kneighbors(q)

    # NOTE: Here I am assuming that the csv file read is the one for which the NN is trained.
    # Otherwise indices will not match.
    for i, a in enumerate(test_authors):
        print('Suggestion for ', a, ': ')
        print(author_df.loc[indices[i]])

        if not os.path.exists(a):
            os.mkdir(a)

        plt.figure(figsize=(24, 20))
        for rank, idx in enumerate(indices[i]):
            name = author_df.loc[idx]['author']
            text = author_df.loc[idx]['doc']
            wordcloud = WordCloud(background_color="white",
                                  height=400, width=600,
                                  max_font_size=100).generate(text)
            plt.subplot(5, 4, rank+1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(name + '(rank: ' + str(rank + 1) + ')', fontsize=25)
            plt.tight_layout()
            # plt.savefig(os.path.join(a, str(rank) + '_' + name + '.png'))
        plt.savefig(a + '_' + model_name + '_sim.png')

def check_lda_model(root_folder):
    lda_model = LDAModel(100, 'full', root_folder)
    lda_model.load()

    assess_model(lda_model, 'lda', distance_metric=jansen_shanon_div, load_pickles=True, root_folder=root_folder)


def check_w2v_model(root_folder):
    w2v_model = Word2vecModel()
    w2v_model.initialize("{}/data/glove.6B/glove.6B.50d.txt".format(root_folder))

    assess_model(w2v_model, 'w2v', distance_metric='cosine', load_pickles=False, root_folder=root_folder)
