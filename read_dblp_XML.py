from DblpXmlHandler import *
import urllib.request
import gzip
import shutil


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/dblp.xml'):
        print('Downloading the dblp xml...')
        urllib.request.urlretrieve('http://dblp.uni-trier.de/xml/dblp.xml.gz', 'data/dblp.xml.gz')
        urllib.request.urlretrieve('http://dblp.uni-trier.de/xml/dblp.dtd', 'data/dblp.dtd')
        print('Extracting the zip file...')
        with gzip.open('data/dblp.xml.gz', 'rb') as f_in, open('data/dblp.xml', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print('Download and extraction finished.')

    dblp_handler = DblpXmlHandler()
    dblp_df = dblp_handler.convert2dataframe('data/dblp.xml')
    dblp_df = dblp_handler.merge_csv_files('tmp')

    dblp_df.to_csv('./data/dblp_all.csv', index=False)