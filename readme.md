# Identify Fraud from Enron Email
Project for Intro to Machine Learning

## Files
* **MP_Proj4_Report.ipynb** The report file describing the project details
* **ML_proj_4.ipynb** The detailed process followed to obtain all the results
* **poi_id.py** poi identifier code: dumps my_classifier.pkl, my_dataset.pkl and my_feature_list.pkl
* **readme.md** this file
* **tester.py** tester code for the poi identifier. NOTE, small changes to the code has been made:
* **requirements.txt** used packages for python

line 15:
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
line 33:
#for train_idx, test_idx in cv:
for train_idx, test_idx in cv.split(features, labels):

## Resources:

- https://scikit-learn.org/
- udacity.com
- wikepedia.com"# UDACITY_Project_4_ML" 
