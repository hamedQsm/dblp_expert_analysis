from DblpXmlHandler import *

if __name__ == '__main__':
    dblp_handler = DblpXmlHandler()
    dblp_df = dblp_handler.convert2dataframe('./data/dblp.xml')
    dblp_df = dblp_handler.merge_csv_files('tmp')

    dblp_df.to_csv('./data/dblp_all.csv', index=False)

    # aggregating by author name, and concatinating the paper titles
    # def aggregate_docs(group):
    #     titles = [str(t) for t in group['title']]
    #     return pd.Series({'title': ' '.join(titles),
    #                       'num': len(group)
    #                       })
    # author_df = dblp_df.groupby('author').apply(aggregate_docs).reset_index()
    # author_df.to_csv('./data/dblp_authors.csv', index=False)
