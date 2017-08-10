from DblpXmlHandler import *

if __name__ == '__main__':
    # read_dblp_xml()
    dblp_handler = DblpXmlHandler()
    dblp_df = dblp_handler.convert2Datafram('./data/dblp.xml', 50000)

    print(dblp_df)
