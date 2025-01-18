Data Splits:
neoplasm_train: 350 abstracts
neoplasm_test: 100 abstracts 
neoplasm_test: 50 abstracts 
glaucoma_test: 100 abstracts 
mixed_test: 100 abstracts (20 on glaucoma, 20 on neoplasm, 20 on diabetes, 20 on hypertension, 20 on hepatitis)

File Format:
The dataset is in the brat format, see below. For each abstract there is one .txt file containing the text and one .ann file containing the annotations. The annotation.conf contains the configuration for the annotations for brat.

Annotation Tool:
The brat annotation tool (version 1.3 "Crunchy Frog") was used for annotation.
It can be downloaded from http://brat.nlplab.org

To visualise the annotations in brat, copy the content of the data file(including the annotation.config) to the data folder in the brat directory, start brat and select the data set in the menu.