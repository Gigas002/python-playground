import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy
from sklearn.metrics import r2_score
import sys
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model
scale = StandardScaler()


def linreg(df):
    # releaseYear + rating = 0.08652565738295831
    # votes       + rating = 0.12764376427136973
    # length      + rating = 0.005967752284430993

    # releaseYear + votes  = 0.041257963979708
    # length      + votes  = 0.08478784936312986

    # releaseYear + length = 0.07948636643216284

    X = df['votes']
    y = df['rating']

    slope, intercept, r, p, std_err = stats.linregress(X, y)

    print(f"r={r}")
    if (r < 0.5):
        print(f"This data doesn't work well with lin")

        return

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, X))

    plt.scatter(X, y)
    plt.plot(X, mymodel)
    plt.show()


def polyreg(df):
    # releaseYear + rating = 0.012568250673094084
    # votes       + rating = 0.020975042538139976
    # length      + rating = 0.04367899313305634

    # releaseYear + votes  = 0.004835948140897717
    # length      + votes  = 0.011643431882585653

    # releaseYear + length = 0.023038718756029697

    X = df['length']
    y = df['rating']

    mymodel = numpy.poly1d(numpy.polyfit(X, y, 3))
    r2 = r2_score(y, mymodel(X))

    print(f"r2={r2}")
    if (r2 < 0.5):
        print(f"This data doesn't work well with poly")

        return

    # specify how the line will display, we start at position 1, and end at position 22:
    myline = numpy.linspace(1, 22, 100)

    plt.scatter(X, y)
    plt.plot(myline, mymodel(myline))
    plt.show()


def multireg(df, is_scaled=True):
    X = df[['length', 'votes']]
    y = df['rating']

    regr = linear_model.LinearRegression()

    if (is_scaled):
        scaledX = scale.fit_transform(X)
        regr.fit(scaledX, y)
        scaled = scale.transform([[110, 1100]])
        pred = regr.predict([scaled[0]])
    else:
        regr.fit(X, y)
        pred = regr.predict([[110, 1100]])

    # print(regr.coef_)
    print(pred)


def dec_tree(df):
    df = df[:2]

    features = ['releaseYear', 'votes', 'length']

    X = df[features]
    y = df['rating']
    y = y.round()

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)

    tree.plot_tree(dtree, feature_names=features)
    plt.show()


def remove_non_number_from_df(df, column_name):
    df[f'{column_name}'] = pd.to_numeric(df[f'{column_name}'], errors='coerce')

    return df.dropna(subset=[f'{column_name}'])


def filter_to_xlsx():
    df = pd.read_csv("./data/mal/user.csv", sep="\t")
    df = df.loc[df['num_completed'] > 1500]

    df.to_excel("./data/mal/user.xlsx")


def linreg_anime(df):
    # valuable params:
    # type, score, scored_by, episodes, members, favorites,
    # total_duration, start_year, start_season

    # members + favorites = 0.784365230915516

    X = df['total_duration']
    y = df['score']

    slope, intercept, r, p, std_err = stats.linregress(X, y)

    print(f"r={r}")
    if (r < 0.5):
        print(f"This data doesn't work well with lin")

        return

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, X))

    plt.scatter(X, y)
    plt.plot(X, mymodel)
    plt.show()


def polyreg_anime(df):
    # valuable params:
    # type, score, scored_by, episodes, members, favorites,
    # total_duration, start_year, start_season

    # favorites + score = 0.5+

    X = df['favorites']
    y = df['score']

    mymodel = numpy.poly1d(numpy.polyfit(X, y, 3))
    r2 = r2_score(y, mymodel(X))

    print(f"r2={r2}")
    if (r2 < 0.5):
        print(f"This data doesn't work well with poly")

        return

    # specify how the line will display, we start at position 1, and end at position 22:
    myline = numpy.linspace(1, 22, 100)

    plt.scatter(X, y)
    plt.plot(myline, mymodel(myline))
    plt.show()


def read_anime():
    df = pd.read_csv("./data/mal/anime.csv")

    # tv special ova ona music movie
    d = {'music': 0, 'movie': 1, 'ona': 2, 'ova': 3, 'special': 4, 'tv': 5}
    df['type'] = df['type'].map(d)
    df = remove_non_number_from_df(df, 'type')
    df = df.loc[df['type'] > 0]
    type = df['type']

    score = df['score']
    score = score.round()
    df = remove_non_number_from_df(df, 'score')
    df = df.loc[df['score'] > 0]

    scored_members = df['scored_by']
    df = remove_non_number_from_df(df, 'scored_by')
    df = df.loc[df['scored_by'] > 0]

    episodes = df['episodes']
    df = remove_non_number_from_df(df, 'episodes')
    df = df.loc[df['episodes'] > 0]

    all_members = df['members']
    df = remove_non_number_from_df(df, 'members')
    df = df.loc[df['members'] > 0]

    favorites = df['favorites']
    df = remove_non_number_from_df(df, 'favorites')
    df = df.loc[df['favorites'] > 0]

    df['total_duration'] = pd.to_timedelta(df['total_duration'])
    total_durations = df['total_duration'].apply(lambda x: x.total_seconds())
    total_durations = total_durations.round()
    df = remove_non_number_from_df(df, 'total_duration')
    df = df.loc[df['total_duration'] > 0]

    start_year = df['start_year']
    df = remove_non_number_from_df(df, 'start_year')
    df = df.loc[df['start_year'] > 0]

    # winter spring summer fall
    d = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    df['start_season'] = df['start_season'].map(d)
    start_season = df['start_season']
    df = remove_non_number_from_df(df, 'start_season')
    df = df.loc[df['start_season'] > 0]

    return df

    # df.to_excel("./data/mal/anime.xlsx")


def multireg_anime(df, is_scaled=False):
    # valuable params:
    # type, score, scored_by, episodes, members, favorites,
    # total_duration, start_year, start_season

    X = df[['type', 'episodes', 'start_year', 'start_season']]
    y = df['score']

    regr = linear_model.LinearRegression()

    if (is_scaled):
        scaledX = scale.fit_transform(X)
        regr.fit(scaledX, y)
        scaled = scale.transform([[110, 1100]])
        pred = regr.predict([scaled[0]])
    else:
        regr.fit(X, y)
        pred = regr.predict([[5, 26, 2023, 1]])

    print(regr.coef_)
    print(pred)


def dec_tree_anime(df):
    # valuable params:
    # type, score, scored_by, episodes, members, favorites,
    # total_duration, start_year, start_season

    # df = df[:2]

    features = ['type']

    X = df[features]
    y = df['score']
    y = y.round()

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)

    tree.plot_tree(dtree, feature_names=features)
    plt.show()


def hierarchical_clustering():
    # Create arrays that resemble two variables in a dataset
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    # Turn the data into a set of points
    data = list(zip(x, y))

    # Compute the linkage between all of the different points
    # Here we use a simple euclidean distance measure and Ward's linkage,
    # which seeks to minimize the variance between clusters
    linkage_data = linkage(data, method='ward', metric='euclidean')

    # Finally, plot the results in a dendrogram
    dendrogram(linkage_data)
    plt.show()


def hierarchical_clustering_sklearn():
    # Create arrays that resemble two variables in a dataset
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    # Turn the data into a set of points
    data = list(zip(x, y))

    # initialize the AgglomerativeClustering class with 2 clusters,
    # using the same euclidean distance and Ward linkage
    hierarchical_cluster = AgglomerativeClustering(
        n_clusters=2, metric='euclidean', linkage='ward')

    # The .fit_predict method can be called on our data to compute the
    # clusters using the defined parameters across our chosen number of clusters
    labels = hierarchical_cluster.fit_predict(data)

    # Finally, if we plot the same data and color the points using the labels assigned
    # to each index by the hierarchical clustering method, we can see the cluster each
    # point was assigned to
    plt.scatter(x, y, c=labels)
    plt.show()


def logit2prob(logr, x):
    # To find the log-odds for each observation, we must first create a formula that
    # looks similar to the one from linear regression, extracting the coefficient and the intercept
    log_odds = logr.coef_ * x + logr.intercept_
    # then convert the log-odds to odds we must exponentiate the log-odds
    odds = numpy.exp(log_odds)
    # now that we have the odds, we can convert it to probability by dividing it by 1 plus the odds
    probability = odds / (1 + odds)

    return (probability)


def logistic_regression():
    # X represents the size of a tumor in centimeters
    # Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work
    X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92,
                    4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)

    # y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes")
    y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # fit() that takes the independent and dependent values as parameters
    # and fills the regression object with data that describes the relationship
    logr = linear_model.LogisticRegression()
    logr.fit(X, y)

    # predict if tumor is cancerous where the size is 3.46mm:
    predicted = logr.predict(numpy.array([3.46]).reshape(-1, 1))


def logreg_anime(df):
    X = df['start_year'].to_numpy()
    X = X.reshape(-1, 1)
    y = df['score'].to_numpy()
    y = y.round()

    logr = linear_model.LogisticRegression()
    logr.fit(X, y)

    predicted = logr.predict(numpy.array([2077]).reshape(-1, 1))


def grid_search_anime(df):
    X = df['start_year'].to_numpy()
    X = X.reshape(-1, 1)
    y = df['score'].to_numpy()
    y = y.round()

    logit = linear_model.LogisticRegression(max_iter=10000)

    C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

    # list to store the score within
    scores = []

    # loop to change out the values of C and evaluate the model with each change
    for choice in C:
        logit.set_params(C=choice)
        logit.fit(X, y)
        scores.append(logit.score(X, y))

    print(scores)


def main():
    df = read_anime()
    grid_search_anime(df)

    return

    df = pd.read_csv("./data/imdb_archive/contentDataPrime.csv")

    # remove trash values
    df = remove_non_number_from_df(df, "releaseYear")
    df = df.loc[df['releaseYear'] > 0]

    df = remove_non_number_from_df(df, "length")
    df = df.loc[df['length'] > 0]

    df = remove_non_number_from_df(df, "votes")
    df = df.loc[df['votes'] >= 0]

    df = remove_non_number_from_df(df, "rating")
    df = df.loc[df['rating'] >= 0]

    dec_tree(df)


main()
