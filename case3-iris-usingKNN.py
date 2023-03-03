from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def KNNAccuracy():

    print("Welcome to Case3 using Knn alogorithm")
    print(" ")

    Iris = load_iris()

    print("Features of iris data set: ")
    print(Iris.feature_names)
    print(" ")
    print("Target of iris data set: ")
    print(Iris.target_names)
    
    print("Iris data set: ")
    for i in range(len(Iris.target)):
        print("ID: %d  Data: %s  Target: %s"%(i,Iris.data[i],Iris.target[i]))
    
    print(" ")
    Data = Iris.data
    Target = Iris.target

    data_train,data_test,target_train,target_test = train_test_split(Data,Target,test_size = 0.5)

    print("Iris data set after splitting and shuffle: ")
    for i in range(len(target_train)):
        print("ID: %d  Data: %s  Target: %s"%(i,data_train[i],target_train[i]))

    Classifier = KNeighborsClassifier()

    classifier = Classifier.fit(data_train,target_train)

    Prediction = Classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,Prediction)

    return Accuracy


def main():

    print("________________________________________________")

    Accuracy = KNNAccuracy()

    print("Accuracy of iris data set model using KNN algorith is: ",Accuracy*100)


if __name__=="__main__":
    main()    