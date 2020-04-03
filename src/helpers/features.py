from plots import scatter, histogram

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