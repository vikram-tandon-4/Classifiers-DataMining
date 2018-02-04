import string
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt




def letter_2_digit_convert(input):
    inputToLower = input.lower()
    alphabetList = list(string.ascii_lowercase)
    characterList = []

    for a in inputToLower:
        characterList.append(alphabetList.index(a) + 1)

    return characterList


def printValue(feature_test,feature_train,y_label_test,y_label_train):
    print("#####################feature_test########################")

    print(feature_test)
    print("#####################feature_train########################")

    print(feature_train)
    print("#####################length of feature_test and train ########################")
    print(np.shape(feature_train))
    print(np.shape(feature_test))

    print(len(feature_train))
    print(len(feature_test))


    print("#####################label test########################")
    print(y_label_test)
    print("#####################label train########################")
    print(y_label_train)
    print("#####################length of label_test and train ########################")

    print(np.shape(y_label_train))
    print(np.shape(y_label_test))

    print(len(y_label_train))
    print(len(y_label_test))

accuracy = []

def calculateAccuracy(feature_test_final,feature_train_final,y_label_test_final,y_label_train_final):

    feature_test_final2 = feature_test_final.transpose()
    feature_train_final2 = feature_train_final.transpose()
    printValue(feature_test_final2, feature_train_final2, y_label_test_final, y_label_train_final)
    # printValue(feature_test_final, feature_train_final,y_label_test_final,y_label_train_final)
    clfCentroid = NearestCentroid()
    clfCentroid.fit(feature_train_final2, y_label_train_final)
    clfCentroid.predict(feature_test_final2)
    acc = clfCentroid.score(feature_test_final2, y_label_test_final)
    print("############### Centroid computed accuracy ############")
    print(clfCentroid.predict(feature_test_final2))
    print(acc)
    accuracy.append(acc)


def  pickData(filename, class_numbers, training_instances, test_instances):

    isFirstTime = True
    feature_train_final = []
    feature_test_final = []
    y_label_train_final = []
    y_label_test_final = []

    file_data = np.genfromtxt(filename, delimiter=",")

    data_of_all_rows = file_data
    array = np.array(data_of_all_rows)
    print("########## Data Retrieved from file ###############")
    print(array)

    total_feature_columns = training_instances + test_instances

    for j in class_numbers:

        start = (j - 1) * total_feature_columns
        end = j * total_feature_columns
        numberOfTestValues = training_instances
        print(start)
        print(end)
        featureLastElement = start + numberOfTestValues
        print(featureLastElement)
        print(end - start)

        feature_train = array[1:, start:featureLastElement]
        feature_test = array[1:, featureLastElement:end]

        y_label_train = array[0, start:featureLastElement]
        y_label_test = array[0, featureLastElement:end]

        print(y_label_test)
        ############# check if first time and initialise values ########################
        if (isFirstTime):
            isFirstTime = False
            feature_train_final = feature_train
            feature_test_final = feature_test
            y_label_train_final = y_label_train
            y_label_test_final = y_label_test


        else:
            feature_train_final = np.hstack((feature_train_final, feature_train))
            feature_test_final = np.hstack((feature_test_final, feature_test))
            y_label_train_final = np.hstack((y_label_train_final, y_label_train))
            y_label_test_final = np.hstack((y_label_test_final, y_label_test))




    calculateAccuracy(feature_test_final, feature_train_final, y_label_test_final, y_label_train_final)
    getTrainData = 'Train Data : \n\n ' + str(feature_train_final) + '\n\nTrain label : \n\n' + str(y_label_train_final)
    text_file = open("trainingData.txt", "w")
    text_file.write(getTrainData)
    text_file.close()



# call HandwrittenLetters
# input = 'abcdefghij'
input = 'qrstuvwxyz'
print(letter_2_digit_convert(input))
characterList = letter_2_digit_convert(input)

trainList = [5,10,15,20,25,30,35]

for i in trainList:
    test = 39-i
    print("################### test and train ####################")
    print(test)
    print(i)
    pickData('HandWrittenLetters.txt', letter_2_digit_convert(input), i, test )


plt.plot(accuracy)
plt.ylabel('accuracy')
plt.show()










