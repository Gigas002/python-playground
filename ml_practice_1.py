# imports

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

def convert_image_to_grayscale(path):
    return Image.open(path).convert("L")

def write_img(grayscale_img, xarr, yarr, insert_id, base_id, data_source):
    flatten_image = grayscale_img.flatten()
    xarr[insert_id] = flatten_image
    yarr[insert_id] = data_source.loc[base_id, "GC"]

def fill_test_data(data_source, data_path):
    # Secure an area to store ndarray data 
    data_len = len(data_source)

    X_data = np.empty((data_len, 480000 ), dtype=np.uint8)
    y_data = np.empty(data_len, dtype=np.uint8)

    # Iterate over each image 
    for i in range(data_len):
        name = data_source.loc[i, "File name"]
        grayscale_img = np.array(convert_image_to_grayscale(f"{data_path}/{name}.jpg"))
        write_img(grayscale_img, X_data, y_data, i, i, data_source)

    return (X_data, y_data)

def fill_data(data_source, data_path):
    data_len = len(data_source)

    # Secure 4 times the capacity to prepare the 180 degrees rotated horizontally and vertically 
    X_data = np.empty((data_len * 4, 480000), dtype=np.uint8)
    y_data = np.empty(data_len * 4, dtype=np.uint8)

    # Iterate over each image 
    for i in range(data_len):
        # read the base image as an ndarray and add it to the training data 
        name = data_source.loc[i, "File name"]

        grayscale_img = np.array(convert_image_to_grayscale(f"{data_path}/{name}.jpg"))
        write_img(grayscale_img, X_data, y_data, i, i, data_source)

        # Add left/right flipped data to training data
        grayscale_img = np.fliplr(grayscale_img)
        write_img(grayscale_img, X_data, y_data, i + data_len, i, data_source)

        # add upside down data to training data
        grayscale_img = np.flipud(grayscale_img)
        write_img(grayscale_img, X_data, y_data, i + data_len * 2, i, data_source)

        # Add 180 degree rotation to training data 
        grayscale_img = np.rot90(grayscale_img, 2)
        write_img(grayscale_img, X_data, y_data, i + data_len * 3, i, data_source)

    return (X_data, y_data)

def main():
    '''
    Main func
    '''

    # get data

    train_data_path = "nagaoka_gc/train"
    test_data_path = "nagaoka_gc/test"

    nagaoka_train_data = pd.read_csv(f"{train_data_path}/train_data.csv")
    nagaoka_test_data = pd.read_csv(f"{test_data_path}/test_data.csv")

    # fill data

    (X_train, y_train) = fill_data(nagaoka_train_data, train_data_path)
    (X_test, y_test) = fill_test_data(nagaoka_test_data, test_data_path)

    # train model

    model = SVC(kernel="linear", gamma="scale", random_state=1)
    model.fit(X_train, y_train)

    # test model

    y_pred = model.predict(X_test)
    print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

main()
