from lxml import etree
import pandas as pd
import time
import os

class DblpXmlHandler:

    def read_dblp_xml(self, dblp_xml_file, out_dir,  doc_per_csv = 1000, total_doc_to_read=5000):
        '''
        This function efficiently read the dblp xml files and stores it as csv chunks.
         These csv files later can be read and merged into one pandas dataframe
         
        :param dblp_xml_file: address to the dblp xml file
        :return: 
        '''

        context = etree.iterparse(dblp_xml_file, load_dtd=True, html=True)

        # this dictionay is going to contain the info regarding a document.
        doc_dic = {}
        doc_dic['author'] = []

        # Extra fields (author and type) to be extracted from the xml for each document
        extract_fields = ['title', 'year']
        document_types = ['www', 'phdthesis', 'inproceedings', 'incollection',
                          'proceedings', 'book', 'mastersthesis', 'article']
        dblp_df = pd.DataFrame(columns=extract_fields + ['author', 'type'])
        row = 0
        csv_count = 0
        document_count = 0
        t = time.time()
        for _, elem in context:
            if elem.tag in extract_fields:
                doc_dic[elem.tag] = elem.text
            if elem.tag == 'author':
                doc_dic['author'].append(elem.text)
            # if this condition as satisfied then one document is read completely
            if elem.tag in document_types:
                doc_dic['type'] = elem.tag
                for author in doc_dic['author']:
                    for key in doc_dic.keys():
                        dblp_df.loc[row, key] = author if key == 'author' else doc_dic[key]

                    row += 1
                doc_dic['author'] = []

                document_count += 1

                # the total read docs is reached doc_per_csv I write it to the dataframe on disk
                # This will avoid the increase in memory usage and hence expedite the reading process
                if document_count % doc_per_csv == 0:
                    print('[', document_count, '] writing the csv file...')
                    print('time between two writing: ', time.time() - t)
                    t = time.time()

                    dblp_df.to_csv(os.path.join(out_dir, 'dblp_' + str(csv_count) + '.csv'), index=False)
                    csv_count += 1
                    row = 0
                    del dblp_df
                    dblp_df = pd.DataFrame(columns=extract_fields + ['author', 'type'])

                if document_count > total_doc_to_read:
                    break

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        del context

    def merge_csv_files(self, dir, file_num = None):
        '''
        reads the already generated csv files and merge them into a final pandas dataframe
        :param dir: the path to csv files
        :param file_num: total files to read if None read all
        :return: the dataframe containing read documents.
        '''
        main_df = pd.DataFrame()
        csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if (os.path.isfile(os.path.join(dir, f)) and f.endswith('csv'))]
        for count, csv_file in enumerate(csv_files):
            if file_num is not None and count > file_num:
                break
            print('[', count, '] reading the csv file...')
            df = pd.read_csv(csv_file)
            main_df = main_df.append(df)
            del df

        return main_df

    def convert2Datafram(self, dblp_xml_file, total_doc_to_read):
        '''
        combine the reading dblp xml into csv chunks and merging them into one data frame
        :param dblp_xml_file: path to the dblp xml file
        :param total_doc_to_read: totol number of documents to retrieve
        :return: the dataframe containing all documents. 
        '''
        tmp_dir = 'tmp'
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
        os.mkdir(tmp_dir)
        self.read_dblp_xml(dblp_xml_file, tmp_dir, total_doc_to_read=total_doc_to_read)
        return self.merge_csv_files(tmp_dir)