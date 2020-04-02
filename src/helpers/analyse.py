from pprint import pprint


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
    print("List of People: ")
    print(', '.join(persons))

    print("")
    print("")
    print("")

    # Let's now find out what kind of data is present for each person, we will take a few samples for the analysis
    print("Some persons data as examples:")
    pprint(data[persons[0]], indent=4)
    pprint(data[persons[-1]], indent=4)

    # Let's find out how many POIs we have on the dataset
    print("")
    print("")
    print("")

    print("Number of persons:", len(persons))
    print("Number of POIs:", len(POIs))
    print("POIs:", ', '.join(POIs))


def fix(data):
    """
    Fixes some of the invalid data points by e.g. performing replacements

    :param data:
    :return:
    """
    # Remove the total
    ds = dict(data)
    del ds['TOTAL']

    # Replace NaN values for zeros
    ff = [
        "salary",
        "deferral_payments",
        "total_payments",
        "loan_advances",
        "bonus",
        "restricted_stock_deferred",
        "deferred_income",
        "total_stock_value",
        "expenses",
        "exercised_stock_options",
        "other",
        "long_term_incentive",
        "restricted_stock",
        "director_fees"
    ]

    for f in ff:
        for person in ds:
            if ds[person][f] == "NaN":
                ds[person][f] = 0

    return ds
