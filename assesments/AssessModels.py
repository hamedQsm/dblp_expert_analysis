import pickle

import pandas as pd
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

from Helper import Helper
from models.LDAModel import LDAModel
from models.w2vecModel import Word2vecModel


def jansen_shanon_div(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


# hamed_papers = \
#     "Cascading failure tolerance of modular small-world networks" + \
#     "Region-Based analysis of hybrid petri nets with a single general one-shot transition" + \
#     "Survivability Evaluation of Gas, Water and Electricity Infrastructures" + \
#     "Survivability evaluation of fluid critical infrastructures using hybrid Petri nets" + \
#     "Energy resilience modelling for smart houses" + \
#     "Survivability analysis of a sewage treatment facility using hybrid Petri nets" + \
#     "Fluid Survival Tool: A Model Checker for Hybrid Petri Nets" + \
#     "Hybrid Petri nets with multiple stochastic transition firings" + \
#     "Approximate Analysis of Hybrid Petri Nets with Probabilistic Timed Transitions" + \
#     "Pricing in population games with semi-rational agents" + \
#     "Analysis of hybrid Petri nets with random discrete events" + \
#     "Influence of short cycles on the PageRank distribution in scale-free random graphs"

test_authors = ['Hamed Ghasemieh', 'Anne Remke', 'Boudewijn R. Haverkort']


def assess_model(model, model_name, data_portion = .7 ,distance_metric=jansen_shanon_div, load_pickles=True):
    '''
    Assess the given model by returning the list of authors 
    :param model_name: 
    :param load_pickles: 
    :param data_portion: 
    :param distance_metric: 
    :param model: 
    :return: 
    '''
    helper = Helper()
    author_df = helper.prepare_author_df('../data/dblp_all.csv', data_portion=data_portion, max_num_pub=1000)
    # author_df = pd.read_csv('../data/{}-authors_processed.csv'.format(data_portion)).head(10000)

    doc_list = [t.split() for t in author_df['doc']]
    print('Total number of processed documents: ', len(doc_list))

    if not load_pickles:
        print('Transforming to LDA space...')
        X = model.transform(doc_list)
        pickle.dump(X, open('../objects/{}-transferred_docs.pickle'.format(model_name), "wb"))

        print('Training the nearest neighbor...')
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree', metric=distance_metric).fit(X)
        pickle.dump(nbrs, open('../objects/{}-nbrs-balltree.pickle'.format(model_name), "wb"))
    else:
        # Loading the pickle in case we have already transformed the data.
        # X = pickle.load(open('../objects/full-{}-transferred_docs.pickle'.format(model_name), "rb"))
        # Loading the pickel in case we have already trained the model.
        nbrs = pickle.load(open('../objects/{}-nbrs-balltree.pickle'.format(model_name), "rb"))

    # starting to query KNN
    query_authors = author_df[author_df['author'].isin(test_authors)]
    print (query_authors)

    doc_list = [(helper.preprocess_text(p).split()) for p in query_authors['doc']]

    q = model.transform(doc_list)
    distances, indices = nbrs.kneighbors(q)

    print(distances)

    # NOTE: Here I am assuming that the csv file read is the one for which the NN is trained.
    # Otherwise indices will not match.
    for i, a in enumerate(test_authors):
        print('Suggestion for ', a, ': ')
        print(author_df.loc[indices[i]])


def check_lda_model():
    lda_model = LDAModel(100, 'full')
    lda_model.load()

    assess_model(lda_model, 'lda', load_pickles=False)

def check_w2v_model():
    w2v_model = Word2vecModel()
    w2v_model.initialize("../data/glove.6B/glove.6B.50d.txt")

    assess_model(w2v_model, 'w2v', load_pickles=False)


if __name__ == '__main__':
    check_lda_model()

