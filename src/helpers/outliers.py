from plots import scatter


def find_outliers(data):
    """
    Finds the outliers in the dataset

    :param data:
    :return:
    """
    scatter(data, ['salary', 'bonus'])

    # Let's find out the possible outliers
    print('Finding possible outliers...')
    potential_outliers = []
    for person in data:
        if data[person]['salary'] > 800000 or data[person]['bonus'] > 6000000:
            potential_outliers.append(person)

    print('Found {:,} potential outliers, "{}"'.format(len(potential_outliers), ', '.join(potential_outliers)))
    # Let's examine now the potential outliers
    outliers = []
    for potential_outlier in potential_outliers:
        if data[potential_outlier]["poi"]:
            print('  -> "{}" is excluded for being a POI'.format(potential_outlier))
        elif data[potential_outlier]['from_poi_to_this_person'] + data[potential_outlier]['from_this_person_to_poi'] > 100:
            print('  -> "{}" is excluded for having high interactions with a POI'.format(potential_outlier))
        else:
            outliers.append(potential_outlier)

    print('Found {:,} actual outliers, "{}"'.format(len(outliers), ', '.join(outliers)))

    return outliers


def remove_outliers(data, outliers):
    """
    Removes the outliers from the dataset

    :param data: Dict with the data associated to each person
    :return: same data without the outliers
    """
    ds = dict(data)
    for outlier in outliers:
        ds.pop(outlier, None)

    print('Removed {:,} outliers from the dataset'.format(len(outliers)))

    return ds