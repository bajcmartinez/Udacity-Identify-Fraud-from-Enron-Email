#!/usr/bin/python
import sys
import os
sys.path.append(os.getcwd()[:os.getcwd().index('src')])

import pickle
from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from helpers.analyse import analyse, fix, try_classifiers
from helpers.outliers import remove_outliers, find_outliers
from helpers.features import create_new_features, find_optimal_features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']
poi_label = ['poi']

new_features = ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio', 'bonus_over_salary_ratio']

features_list = poi_label + financial_features + email_features  # You will need to use more features

### Load the dictionary containing the dataset
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Debug the data to have a better understanding on the dataset
analyse(data_dict)

### Next we fix some of the identified issues
data_dict = fix(data_dict)

### Task 2: Remove outliers
outliers = find_outliers(data_dict)
my_dataset = remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = create_new_features(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
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

# Though is not the best idea, let's run the classifiers against the full features list
# try_classifiers(my_dataset, features_list)

# Now let's try to move from there to finding the best features for the task
optimal_features_list = find_optimal_features(my_dataset, features_list)
clf = try_classifiers(my_dataset, optimal_features_list)

### Final result:
# Trying AdaBoost...
# accuracy: 0.83
# precision: 0.332838383838
# recall:    0.3155

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, optimal_features_list)
