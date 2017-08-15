from DblpXmlHandler import *

if __name__ == '__main__':
    dblp_handler = DblpXmlHandler()
    dblp_df = dblp_handler.convert2dataframe('data/dblp.xml')
    dblp_df = dblp_handler.merge_csv_files('tmp')

    dblp_df.to_csv('./data/dblp_all.csv', index=False)