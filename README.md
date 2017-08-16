# DBLP Expert Analysis
The goal is to the level of expertise authors on different (inferred) topics from dblp paper database.

## Content

### read_dbpl_XML.py
Downloads (if the dblp.xml is not already in data frame) and reads the dblb.xml, and stores it as csv file.

### DblpXmlHandler.py
Contains the main functionality to efficiently parsing the xml file, without loading it into memory completely.
It can return a pandas data frame.

### Helper.py
Include the needed function for preprocessing the pandas data frame of authors.
This can create a pandas data frame with each row containing an author with all his/her publication processed and
concatenated in one piece of text.

### models/LDAModel.py
The main class for training, loading and storing the LDA model.
This also include function for transforming a new document to LDA topic space.

### models/w2vModel.py
The main class for loading the already trained w2v model plus transferring new document to w2v space.

### assessments/assessModels.py
Includes the code for assessing both LDA and w2v models.
Using the K nearest Neighbors algorithm it finds the closest authors to given authors.
This also generates word cloud for closest authors given.

### assessments/visualize_w2v.py
Generates a K-means clustering on the first two PCs of vector space created by w2v.
I wanted to used this for a prototype of a more advanced algorithm.

### trainLDA.py
Starts the training process for LDA

### test.py
It just call check_lda_model/check_w2v_model from assessModels.py




