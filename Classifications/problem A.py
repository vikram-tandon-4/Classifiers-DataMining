
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold # import KFold
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt

# Neighbors taken are 5
clfKNN = KNeighborsClassifier(n_neighbors=5)
clfSVM = svm.SVC()
clfCentroid = NearestCentroid()


file_data = np.genfromtxt('ATNTFaceImages400.txt', delimiter=",")

# Transpose given data
data_of_all_rows = file_data
array = np.array(data_of_all_rows)
data = array.transpose()

print("########## Data Retrieved from file ###############")
print(data)

labels_test = data[0:,0]
labels_train = data[0:,0]

print("################ Labels train ############################")
print(labels_train)

print("################ Number of labels ############################")
number_of_labels = len(labels_train)
print(number_of_labels)

print("################## Features  total  data ##############################")

total_column_data = data[0:, 1:]
print(total_column_data)


################## Kfold method ############################
X = total_column_data

print("########## printing X ######################")
print(X)
y = labels_train

knn =[]
def calculateKnnAccuracy(kFold):
    total = 0
    accuracy = []

    kf = KFold(n_splits=kFold, random_state=None, shuffle=True)
    print("printing kf", kf)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ########################## Train Data for KNN  #################################
        clfKNN.fit(X_train, y_train)
        clfKNN.predict(X_test)
        acc = clfKNN.score(X_test, y_test)
        total += acc
        accuracy.append(acc)
        print("############### KNN computed accuracy on each iteration############")
        print(acc)


    print("############### Average #################")
    print(total / kFold)
    knn.append(total / kFold)
    print('########### accuracy list############')
    print(accuracy)


svm =[]



def calculateSvmAccuracy(kFold):

    total = 0
    accuracy = []

    kf = KFold(n_splits=kFold, random_state=None, shuffle=True)
    print("printing kf", kf)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ########################## Train Data for SVM  #################################

        clfSVM.fit(X_train, y_train)
        clfSVM.predict(X_test)
        acc = clfSVM.score(X_test, y_test)
        total += acc

        print("############### SVM computed accuracy on each iteration############")
        print(acc)
        accuracy.append(acc)

    print("############### Average #################")
    print(total / kFold)
    svm.append(total / kFold)
    print('########### accuracy list############')
    print(accuracy)



centroid =[]
def calculateCentroidAccuracy(kFold):

    total=0
    accuracy = []
    kf = KFold(n_splits=kFold, random_state=None, shuffle=True)
    print("printing kf", kf)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ########################## Train Data for Centroid  #################################
        clfCentroid.fit(X_train, y_train)
        clfCentroid.predict(X_test)
        acc = clfCentroid.score(X_test, y_test)
        print("############### Centroid computed accuracy on each iteration############")
        total += acc
        print(acc)
        accuracy.append(acc)

    print("############### Average #################")
    print(total / kFold)
    centroid.append(total / kFold)
    print('########### accuracy list############')
    print(accuracy)

cvv =[2,3,5,10]


for i in cvv:
    calculateKnnAccuracy(i)
    calculateSvmAccuracy(i)
    calculateCentroidAccuracy(i)

print('################# SVM points #############')
print(svm)
print('################# KNN points #############')
print(knn)
print('################# Centroid points #############')
print(centroid)

plt.plot(knn, 'r--',  centroid, 'b--',  svm, 'g--')
plt.show()