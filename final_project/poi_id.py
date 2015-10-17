#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### feature selection using SelectKBest

def select_best_features(data_dict, features_list, num):    
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    k_best = SelectKBest(k=num)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    best_features_dict = dict(sorted_pairs[:num])
    best_features=['poi']
    best_features+=best_features_dict.keys()
    return best_features


### this function is used to create new feature for each person
### representing sum of both total_payments and total_stock_value

def add_total_payments_stock(data_dict, features_list):
    features = ['total_stock_value','total_payments']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for feature in features:
            if person[feature] == 'NaN':
                is_valid = False
        if is_valid:      
            total_payments_stock = person['total_stock_value']+person['total_payments']
            person['total_payments_stock'] = total_payments_stock
        else:
            person['total_payments_stock'] = 'NaN'
    features_list += ['total_payments_stock']

# all the available features
all_features_list = ['bonus',					 
					 #'loan_advances',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'exercised_stock_options',
                     'expenses',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'to_messages']

# You will need to use more features

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.

my_dataset = data_dict

features_list=select_best_features(data_dict, all_features_list, 5)

add_total_payments_stock(data_dict, features_list)

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#naive_bayes classifier

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    

"""
SVM classifier
from sklearn.svm import SVC
clf = SVC(C=1000., gamma=0.01)
"""

"""
from sklearn import tree
_clf = tree.DecisionTreeClassifier(min_samples_split=100)

from sklearn.grid_search import GridSearchCV
params = { "min_samples_split":[10, 30, 50, 70, 100],
                   "criterion": ('gini', 'entropy')
                    }
clf = GridSearchCV(_clf, params)
clf = clf.fit(features, labels)
print clf.best_estimator_
"""
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)
#print clf.feature_importances_
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)