from plots import scatter, histogram
from pprint import pprint
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from src.tools.feature_format import featureFormat, targetFeatureSplit


def find_optimal_features(data, features_list):
    """

    :param data:
    :param features_list
    :return:
    """
    print('Finding optimal features')
    ds = featureFormat(data, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(ds)

    # Set up the scaler
    minmax_scaler = preprocessing.MinMaxScaler()
    features_minmax = minmax_scaler.fit_transform(features)

    # Apply SelectKBest
    k_best = SelectKBest(chi2, k=10)

    # Use the instance to extract the k best features
    k_best.fit_transform(features_minmax, labels)

    feature_scores = ['%.2f' % elem for elem in k_best.scores_]
    feature_scores_pvalues = ['%.3f' % elem for elem in k_best.pvalues_]
    k_indices = k_best.get_support(indices=True)

    k_features = [(features_list[i + 1],
                   feature_scores[i],
                   feature_scores_pvalues[i]) for i in k_indices]

    print('Optimal features:')
    pprint(k_features)

    optimal = [features_list[i + 1] for i in k_indices]
    # If poi was not selected, let's manually add it, I'd like to keep it
    if 'poi' not in optimal:
        optimal.insert(0, 'poi')

    return optimal


def create_new_features(data):
    """
    Adds additional features to the dataset

    :param data:
    :return:
    """
    print("Adding features...")
    for person in data:
        from_poi_to_this_person = data[person]['from_poi_to_this_person']
        to_messages = data[person]['to_messages']
        from_messages = data[person]['from_messages']
        from_this_person_to_poi = data[person]['from_this_person_to_poi']

        salary = data[person]['salary']
        bonus = data[person]['bonus']

        # Let's add some interesting ratios
        data[person]['from_poi_to_this_person_ratio'] = from_poi_to_this_person / float(to_messages)
        data[person]['from_this_person_to_poi_ratio'] = from_this_person_to_poi / float(from_messages)

        if salary != 0:
            data[person]['bonus_over_salary_ratio'] = bonus / float(salary)
        else:
            data[person]['bonus_over_salary_ratio'] = 0

    # Let's render some charts to help us understand this new features
    scatter(data, ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio'])
    histogram(data, 'bonus_over_salary_ratio')

    return data