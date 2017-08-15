from Helper import Helper
from models.LDAModel import LDAModel
import pandas as pd

if __name__ == '__main__':
    helper = Helper()
    # author_df = helper.prepare_author_df('data/dblp_all.csv', data_portion=.7, max_num_pub=1000, root_folder='.')
    author_df = pd.read_csv('data/0.7-authors_processed.csv')

    doc_list = [t.split() for t in author_df['doc']]
    print('Total number of processed documents: ', len(doc_list))
    #
    lda_model = LDAModel(50, '50-topics', '.')
    # lda_model.load()
    lda_model.initialize(doc_list)
    lda_model.fit()

    print ('Starting visualization...')
    lda_model.visualize()