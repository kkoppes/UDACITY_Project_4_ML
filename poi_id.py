#!/usr/bin/python

import sys
import pickle

from math import isnan
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def remove_keys(dictionary, remove_list):
    """
    remove entry in dicrionary by key
    """
    for key in remove_list:
        try:
            del dictionary[key]
        except:
            print("error : " + str(key) + " not in dictionary" )
    return dictionary


def replace_nan_ndict(ndict):
    """
    replace NaN with 0 in the dictionary
    """
    for k,v in ndict.items():
        for key, value in v.items():
            if isinstance(value, (int, long, float, complex)) and not isinstance(value, bool):
                if isnan(value) == True:
                    ndict[k][key] = 0
            elif value == 'NaN':
                ndict[k][key] = 0
    return ndict


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'from_messages',
                 'restricted_stock_deferred',
                 'to_messages',
                 'director_fees',
                 'other',
                 'from_poi_to_this_person',
                 'expenses',
                 'loan_advances',
                 'restricted_stock',
                 'total_payments',
                 'deferred_income',
                 'salary',
                 'bonus',
                 'exercised_stock_options',
                 'total_stock_value',
                 'deferral_payments_r',
                 'shared_receipt_with_poi_r',
                 'long_term_incentive_r',
                 'from_this_person_to_poi_r']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)

### Task 2: Remove outliers
non_persons = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
enron_data = remove_keys(enron_data, non_persons)
# fix transposed date points
enron_data['BHATNAGAR SANJAY'] = {'salary': 0,
                                 'to_messages': 523,
                                 'total_stock_value': 15456290,
                                 'deferral_payments': 0,
                                 'total_payments': 137864,
                                 'loan_advances': 0,
                                 'bonus': 0,
                                 'email_address': 'sanjay.bhatnagar@enron.com',
                                 'restricted_stock_deferred': -2604490,
                                 'deferred_income': 0,
                                 'expenses': 137864,
                                 'from_poi_to_this_person': 0,
                                 'exercised_stock_options': 15456290,
                                 'from_messages': 29,
                                 'other': 0,
                                 'from_this_person_to_poi': 1,
                                 'poi': False,
                                 'long_term_incentive': 0,
                                 'shared_receipt_with_poi': 463,
                                 'restricted_stock': 2604490,
                                 'director_fees': 0}

enron_data['BELFER ROBERT'] = {'salary': 0,
                              'to_messages': 0,
                              'deferral_payments': 0,
                              'total_payments': 3285,
                              'loan_advances': 0,
                              'bonus': 0,
                              'email_address': 0,
                              'restricted_stock_deferred': -44093,
                              'deferred_income': -102500,
                              'total_stock_value': 0,
                              'expenses': 3285,
                              'from_poi_to_this_person': 0,
                              'exercised_stock_options': 0,
                              'from_messages': 0,
                              'other': 0,
                              'from_this_person_to_poi': 0,
                              'poi': False,
                              'long_term_incentive': 0,
                              'shared_receipt_with_poi': 0,
                              'restricted_stock': 44093,
                              'director_fees': 102500}
# remove Eugene, since there is no info on him.
enron_data = remove_keys(enron_data, ['LOCKHART EUGENE E'])

### Task 3: Create new feature(s)
for name, values in enron_data.items():
    try:
        enron_data[name]['deferral_payments_r'] = float(enron_data[name][
            'deferral_payments']) / float(enron_data[name]['total_payments'])
    except ZeroDivisionError:
        enron_data[name]['deferral_payments_r'] = 0

    try:
        enron_data[name]['long_term_incentive_r'] = float(enron_data[name][
            'long_term_incentive']) / float(enron_data[name]['total_payments'])
    except ZeroDivisionError:
        enron_data[name]['long_term_incentive_r'] = 0

    try:
        enron_data[name]['shared_receipt_with_poi_r'] = float(enron_data[name][
            'shared_receipt_with_poi']) / float(enron_data[name]['to_messages'])
    except ZeroDivisionError:
        enron_data[name]['shared_receipt_with_poi_r'] = 0

    try:
        enron_data[name]['from_this_person_to_poi_r'] = float(enron_data[name][
            'from_this_person_to_poi']) / float(enron_data[name]['from_messages'])
    except ZeroDivisionError:
        enron_data[name]['from_this_person_to_poi_r'] = 0

### Store to my_dataset for easy export below. make sure no nan are in the data,
### otherwise the tester is crashing
my_dataset = replace_nan_ndict(enron_data)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC
"""
Please see the ML_proj_4.ipynb for the selection process between different
Classifiers and scalers
"""
clf = Pipeline([('scaler', QuantileTransformer(output_distribution='uniform')),
                ('SVC',SVC(kernel='poly',
                           C=50))])

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
"""
Please see the ML_proj_4.ipynb for the tuning process of the different
Classifiers
"""

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
