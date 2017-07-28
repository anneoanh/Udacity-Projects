#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
import pprint as p
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    'poi',
    'salary', 
    'deferral_payments', 
    'loan_advances', 
    'bonus', 
    'restricted_stock_deferred', 
    'deferred_income', 
    'expenses', 
    'exercised_stock_options', 
    'other', 
    'long_term_incentive', 
    'restricted_stock', 
    'director_fees',
    'to_messages', 
    'from_poi_to_this_person', 
    'from_messages', 
    'from_this_person_to_poi', 
    'shared_receipt_with_poi',
    'fraction_from_poi',
    'fraction_to_poi'
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# p.pprint(data_dict)
### Task 2: Remove outliers
from tools import outlier_cleaner
outliers = {
    "TOTAL", 
    "THE TRAVEL AGENCY IN THE PARK", 
    "LOCKHART EUGENE E",
    "GRAMM WENDY L,"
    "SCRIMSHAW MATTHEW",
    "WHALEY DAVID A",
    "WODRASKA JOHN",
    "WROBEL BRUCE",
    "CHRISTODOULOU DIOMEDES",
    "CLINE KENNETH W",
    "GILLIS JOHN",
    "SAVAGE FRANK",
    "WAKEHAM JOHN",
    "LAVORATO JOHN J",
    "ALLEN PHILLIP K",
    "KITCHEN LOUISE",
    "WHALLEY LAWRENCE G",
    "MCMAHON JEFFREY",
    "FALLON JAMES B",
    "SHANKMAN JEFFREY A",
    "HICKERSON GARY J",   
    "SHERRIFF JOHN R"
}
outlier_cleaner(data_dict, outliers)

### Task 3: Create new feature(s)
from tools import computeFraction
for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from time import time
from tools import custom_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer
from tester import test_classifier

# Select and scale features
scaler = MinMaxScaler()
kbest = SelectKBest()


# Combine features
combined_features = FeatureUnion([
    # ("SCALER", scaler),
    ("KBEST", kbest)
])

# Create classifier
dt = DecisionTreeClassifier()

# Create pipeline
pipeline = Pipeline([
    ("FEATURES", combined_features),
    ("DT", dt)
])

# Create cross validation 
cv = StratifiedShuffleSplit(10, random_state=42)

print "Initiate grid search..."
t0 = time()

# Define tuning parameters
params = {
    "FEATURES__KBEST__k": range(1,19),
    "DT__max_depth": range(2,10),
    "DT__min_samples_split": range(2,10),
    "DT__min_samples_leaf": range(2,5),
    "DT__criterion": ["entropy", "gini"],
    "DT__splitter": ['best', "random"]
}

# Create grid search
gs = GridSearchCV(pipeline, param_grid=params, cv=cv, scoring=make_scorer(custom_score))
gs.fit(features, labels)

print "Total time:", round(time()-t0, 3), "s"
print gs.best_params_
# Assign classifier with best estimator
clf = gs.best_estimator_
print "\nDecistion Tree test results:"
test_classifier(clf, my_dataset, features_list)
print "\nDone."

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
