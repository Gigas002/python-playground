from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

irisdata = datasets.load_iris()

# print some info about dataset

# print(irisdata.DESCR)

# print("Dataset's data:")
# print(irisdata.data)
# print("Dataset's data's shape:")
# print(irisdata.data.shape)
# print("Dataset's target:")
# print(irisdata.target)

# print(irisdata.feature_names)
# print(irisdata.target_names)

# split dataset into train data and test data randomly
# train size is 80% of dataset and test data is 20% of dataset

X_train, X_test, y_train, y_test = train_test_split(irisdata.data, irisdata.target, test_size = 0.2 , train_size = 0.8 , shuffle = True )

# train model

classifier = SVC(kernel = "linear", gamma = "scale")
classifier.fit(X_train, y_train)

# test the model

y_pred = classifier.predict(X_test)
print(y_pred)

# compare with actual test data

# print(y_test)

# calculate the accurancy score

print(metrics.accuracy_score(y_test, y_pred))

# show the confusion matrix

print(metrics.confusion_matrix(y_test, y_pred))

# show training report

print(metrics.classification_report(y_test, y_pred))
