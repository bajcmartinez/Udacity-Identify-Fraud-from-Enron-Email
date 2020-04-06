from pprint import pprint
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from src.tools.feature_format import featureFormat, targetFeatureSplit
from src.tester import test_classifier
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from numpy import mean


def analyse(data):
    """
    Let's take a deeper look into the data

    :param data: Dict with the data associated to each person
    :return:
    """
    persons = data.keys()
    POIs = filter(lambda person: data[person]['poi'], data)

    # First let's identify all t
    # he keys present in the dictionary, which are the persons available on the DS
    print('List of People: ')
    print(', '.join(persons))

    print('')
    print('')
    print('')

    # Let's now find out what kind of data is present for each person, we will take a few samples for the analysis
    print('Some persons data as examples:')
    print('{}:'.format(persons[0]))
    pprint(data[persons[0]], indent=4)

    print('{}:'.format(persons[-1]))
    pprint(data[persons[-1]], indent=4)

    # Let's find out how many POIs we have on the dataset
    print('')
    print('')
    print('')

    print('Number of persons:', len(persons))
    print('Number of POIs:', len(POIs))
    print('POIs:', ', '.join(POIs))


def fix(data):
    """
    Fixes some of the invalid data points by e.g. performing replacements

    :param data:
    :return:
    """
    # Replace NaN values for zeros
    ff = [
        'salary',
        'deferral_payments',
        'total_payments',
        'loan_advances',
        'bonus',
        'restricted_stock_deferred',
        'deferred_income',
        'total_stock_value',
        'expenses',
        'exercised_stock_options',
        'other',
        'long_term_incentive',
        'restricted_stock',
        'director_fees',
        'from_poi_to_this_person',
        'from_this_person_to_poi'
    ]

    for f in ff:
        for person in data:
            if data[person][f] == 'NaN':
                data[person][f] = 0

    return data


def evaluate_clf(grid_search, features, labels, params, iters=100):
    """
    Evaluate a classifier

    :param grid_search:
    :param features:
    :param labels:
    :param params:
    :param iters:
    :return:
    """
    acc = []
    pre = []
    recall = []

    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)]
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


def try_classifiers(data, features_list, tune=False):
    """
    Try a variety of classifiers for the given features

    :param data:
    :param features_list
    :param tune:
    :return:
    """
    print('Trying classifiers for features: [{}]'.format(', '.join(features_list)))
    # ds = featureFormat(data, features_list, sort_keys=True)
    # labels, features = targetFeatureSplit(ds)

    print('Trying GaussianNB...')
    clf_gb = GaussianNB()
    clf_gb_grid_search = GridSearchCV(clf_gb, {})
    test_classifier(clf_gb_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    print('Trying AdaBoost...')
    clf_ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                                n_estimators=50, learning_rate=.8)
    clf_ab_grid_search = GridSearchCV(clf_ab, {})
    test_classifier(clf_ab_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    if tune:
        print('Trying Tuned AdaBoost...')
        clf_abt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                                    n_estimators=50, learning_rate=.8)
        clf_abt_grid_search = GridSearchCV(clf_abt, {'n_estimators': [1, 5, 10, 50]})
        test_classifier(clf_abt_grid_search, data, features_list)
        # evaluate_clf(nb_grid_search, features, labels, {})

    print('Trying RandomForest...')
    clf_rf = RandomForestClassifier()
    clf_rf_grid_search = GridSearchCV(clf_rf, {})
    test_classifier(clf_rf_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    if tune:
        print('Trying Tuned RandomForest...')
        clf_rft = RandomForestClassifier()
        clf_rft_grid_search = GridSearchCV(clf_rft, {'n_estimators': [4, 5, 10, 500]})
        test_classifier(clf_rft_grid_search, data, features_list)
        # evaluate_clf(nb_grid_search, features, labels, {})

    print('Trying SVC...')
    clf_svc = SVC(kernel='linear', max_iter=1000)
    clf_svc_grid_search = GridSearchCV(clf_svc, {})
    test_classifier(clf_svc_grid_search, data, features_list)
    # evaluate_clf(nb_grid_search, features, labels, {})

    if tune:
        print('Trying Tuned SVC...')
        param_grid = {'C': [1, 50, 100, 1000],
                      'gamma': [0.5, 0.1, 0.01],
                      'degree': [1, 2],
                      'kernel': ['rbf', 'poly', 'linear'],
                      'max_iter': [1, 100, 1000]}
        clf_svct = SVC(kernel='linear', max_iter=1000)
        clf_svct_grid_search = GridSearchCV(clf_svct, param_grid)
        test_classifier(clf_svct_grid_search, data, features_list)
        # evaluate_clf(nb_grid_search, features, labels, {})

    # Return the one which perform the best
    return clf_ab_grid_search
