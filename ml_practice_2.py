# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics

def get_population_data(data, years):
    # create empty arrays for {years} data
    X = np.empty((years, 1), dtype=np.uint32)
    y = np.empty((years, 1), dtype=np.uint32)

    # fill it with tokyo data
    for i in range(years):
        X[i][0] = data.iloc[i, 3]
        y[i][0] = data.iloc[i + 1, 3]

    return (X, y)

def tokyo_population_model(data, years):
    # query for tokyo using column index instead of name
    tokyo_idx = 13
    tokyo = data[data.iloc[:, 1] == tokyo_idx]

    # get train data and separate it with test data

    (X, y) = get_population_data(tokyo, years)

    X_train = X[:50]
    y_train = y[:50]
    X_test = X[50:]
    y_test = y[50:]

    # create model

    model = LinearRegression()
    model.fit(X_train, y_train)

    # check the model

    y_pred = model.predict(X_test).astype(np.uint32)
    # y_pred_gr = np.concatenate([y_train, y_pred])

    # Graph display of correct and predicted values 
    plt.plot(range(8), y_pred, label= 'Predicted' , color= 'red' )
    plt.plot(range(8), y_test, label= 'Actual' , color= 'blue' )
    plt.xlabel('Years')
    plt.ylabel('Population')
    plt.title("Tokyo's population")
    plt.grid(True)
    plt.legend(loc = "upper left")

    plt.show()

def create_prefecture_matrix(data, years, pref_count):
    matrix_size = pref_count + 1
    X = np.zeros((years * matrix_size, matrix_size), dtype=np.uint32)
    y = np.zeros(years * matrix_size, dtype=np.uint32)

    counter = 0
    for i in range(years * matrix_size):
        pref_id = data.iloc[i, 1]
        population = data.iloc[i, 3]
        next_population = data.iloc[i + pref_count + 1, 3]

        # 14 and 8 are kanagawa and ibaraki indexes
        if pref_id < 14:
            X[counter][pref_id - 8] = 1
            
        X[counter][6] = population
        y[counter] = next_population
        counter += 1

    return (X, y)

def kanto_population_model(data, years):
    ibaraki_idx = 8
    kanagawa_idx = 14

    # query for kanto prefs using column index instead of name
    kanto = data[(data.iloc[:, 1] >= ibaraki_idx) & (data.iloc[:, 1] <= kanagawa_idx)]

    (X, y) = create_prefecture_matrix(kanto, years, kanagawa_idx - ibaraki_idx)

    # Split data from 1960 to 2009 as training data, # data after 2010 as test data 
    X_train = X[:350]
    y_train = y[:350]
    X_test = X[350:]
    y_test = y[350:]

    # create and test linear regression model

    l_model = LinearRegression(n_jobs=-1)
    l_model.fit(X_train, y_train)
    l_pred = l_model.predict(X_test).astype(np.uint32)

    # create and test random forest model

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test).astype(np.uint32)

    # create and test SVR model

    svr_model = SVR(gamma='scale')
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test).astype(np.uint32)

    # print(svr_pred)

def main():
    data = pd.read_csv("japan_population_3.csv")
    years = 58

    # tokyo_population_model(data, years)
    kanto_population_model(data, years)

main()
