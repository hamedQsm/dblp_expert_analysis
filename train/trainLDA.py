from Helper import Helper
from models.LDAModel import LDAModel

if __name__ == '__main__':
    helper = Helper()
    author_df = helper.prepare_author_df('../data/dblp_all.csv', data_portion=.4, max_num_pub=1000)

    doc_list = [t.split() for t in author_df['doc']]
    print('Total number of processed documents: ', len(doc_list))

    lda_model = LDAModel(100, 'half')
    # lda_model.load()
    lda_model.initialize(doc_list)
    lda_model.fit()

    print ('Starting visualization...')
    lda_model.visualize()