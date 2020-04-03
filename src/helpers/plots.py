from src.tools.feature_format import featureFormat
import matplotlib.pyplot as plt


def scatter(data, features):
    """
    Renders a scatter plot based on the provided features and data

    :param data:
    :param features:
    :return: rendered points
    """
    points = featureFormat(data, features)

    # Now let's plot this points
    for point in points:
        plt.scatter(point[0], point[1])

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()

    return points


def histogram(data, feature):
    """
    Renders a scatter plot based on the provided features and data

    :param feature:
    :param data:
    :return: rendered points
    """
    points = featureFormat(data, [feature])

    plt.hist(points)
    plt.xlabel(feature)
    plt.show()

    return points
