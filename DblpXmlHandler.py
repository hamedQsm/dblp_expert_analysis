from lxml import etree
import pandas as pd
import time
import os
import shutil
import re

class DblpXmlHandler:

    def read_dblp_xml(self, dblp_xml_file, out_dir,  doc_per_csv = 1000, total_doc_to_read=5000, skip_doc_num=0):
        '''
        This function efficiently read the dblp xml files and stores it as csv chunks.
         These csv files later can be read and merged into one pandas dataframe
         
        :param skip_doc_num: Defines how many of doc to escape (in case they are already read and stored as csv)
        :param dblp_xml_file: address to the dblp xml file
        :return: 
        '''

        context = etree.iterparse(dblp_xml_file, load_dtd=True, html=True)

        # this dictionay is going to contain the info regarding a document.
        doc_dic = {}
        doc_dic['author'] = []

        # Extra fields (author and type) to be extracted from the xml for each document
        extract_fields = ['title', 'year']

        # End tags determining the type of docs in dblp xml
        document_types = ['www', 'phdthesis', 'inproceedings', 'incollection',
                          'proceedings', 'book', 'mastersthesis', 'article']
        dblp_df = pd.DataFrame(columns=extract_fields + ['author', 'type'])

        # row of the dataframe to append the doc to
        row = 0

        # tracks the number of csv files stored
        csv_count = 0

        # tracks the number of docs read
        document_count = 0

        t = time.time()
        for _, elem in context:
            if elem.tag in extract_fields:
                doc_dic[elem.tag] = elem.text
            if elem.tag == 'author':
                doc_dic['author'].append(elem.text)
            # if this condition as satisfied then one document is read completely
            if elem.tag in document_types:
                document_count += 1

                # skipping the number of docs specified as input argument.
                if document_count > skip_doc_num:
                    doc_dic['type'] = elem.tag
                    for author in doc_dic['author']:
                        for key in doc_dic.keys():
                            dblp_df.loc[row, key] = author if key == 'author' else doc_dic[key]

                        row += 1
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

                    if total_doc_to_read is not None and document_count > total_doc_to_read:
                        break
                else:
                    if document_count % doc_per_csv == 0:
                        csv_count += 1
                        print('skipped docs: ', document_count, ' | skipped csv: ', csv_count)

                # we have to empty author array for the next document.
                del doc_dic['author']
                doc_dic['author'] = []


            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        del context

    def merge_csv_files(self, dir, file_num = None):
        '''
        reads the already generated csv files and merge them into a final pandas dataframe
        :param dir: the path to csv files
        :param file_num: total files to read if None read all
        :return: the dataframe containing with each row containing the an author, its publication title, 
                 year, and type of publication.
        '''
        main_df = pd.DataFrame()
        csv_files = [os.path.join(dir, f) for f in os.listdir(dir)
                     if (os.path.isfile(os.path.join(dir, f)) and f.endswith('csv'))]
        for count, csv_file in enumerate(csv_files):
            if file_num is not None and count > file_num:
                break
            print('[', count, '] reading the csv file...')
            df = pd.read_csv(csv_file)
            main_df = main_df.append(df)
            del df

        return main_df

    def convert2dataframe(self, dblp_xml_file, total_doc_to_read=None, csv_dir='tmp', keep_csvs=True, doc_per_csv=1000):
        '''
        combine the reading dblp xml into csv chunks and merging them into one data frame
        :param doc_per_csv: How many docs to store in each csv chunk. 
               This is needed to be passed in case we want to use already stored csv chunks
        :param keep_csvs: path to the directory for storing csv chunks
        :param csv_dir: do we keep already read csv chunks?
        :param dblp_xml_file: path to the dblp xml file
        :param total_doc_to_read: totol number of documents to retrieve
        :return: the dataframe containing with each row containing the an author, its publication title, 
                 year, and type of publication. 
        '''

        skip_doc_num = 0
        if os.path.exists(csv_dir):
            if not keep_csvs:
                shutil.rmtree(csv_dir)
            else:
                csv_list = [f for f in os.listdir(csv_dir)
                            if (os.path.isfile(os.path.join(csv_dir, f)) and f.endswith('csv'))]
                last_csv_num = max(
                    [int(0 if len(re.findall(r'\d+', f)) == 0 else re.findall(r'\d+', f)[0]) for f in csv_list]
                )

                # note that we already need to know how many docs are in each csv stored
                # (this is different from row num of csv's since one doc can have multiple author)
                skip_doc_num = last_csv_num * doc_per_csv
        else:
            os.mkdir(csv_dir)

        print('Starting to read documents...')
        print('Skipping ', skip_doc_num  ,' documents...')
        self.read_dblp_xml(dblp_xml_file, csv_dir, total_doc_to_read=total_doc_to_read,
                           skip_doc_num=skip_doc_num, doc_per_csv=doc_per_csv)
        return self.merge_csv_files(csv_dir)