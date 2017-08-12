from Helper import Helper
from LDAModel import LDAModel

if __name__ == '__main__':
    helper = Helper()
    # doc_list = helper.prepare_author_df('./data/dblp_mil.csv')
    #
    lda_model = LDAModel()
    # lda_model.initialize(doc_list)

    lda_model.train(num_topics=50)

    lda_model.visualize()