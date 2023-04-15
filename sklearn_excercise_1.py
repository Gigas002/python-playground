from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics

# init dataset

digits = datasets.load_digits()


# take the data and target

X = digits.data
y = digits.target

# some checks

# print(digits.DESCR)

# print(X.shape)
# print(y.shape)

# print(X[0])
# print(y[0:50])

# fig = plt.figure()

# for i, x in enumerate(X[0:10], 0):
#     sp = fig.add_subplot(2, 5, (i + 1))
#     if i <= 0:
#         print(x.reshape(8, 8))
#     sp.imshow(x.reshape(8, 8), cmap = "gray")

# plt.show()

# separate data on train and test

X_train = X[:1201] 
y_train = y[:1201] 
X_test = X[1201:] 
y_test = y[1201:]

# generate model

classifier = SVC(kernel="linear", gamma="scale")
classifier.fit(X_train, y_train)

# check the model

y_pred = classifier.predict(X_test)
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
