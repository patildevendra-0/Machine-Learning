
from sklearn.metrics import accuracy_score
from sklearn import tree 
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def Accuracy_usingDecisionTree():

    print("Welcome Case4: Accuracy using decision tree algorithm")
    print(" ")

    Iris = load_iris()

    Data = Iris.data
    Target = Iris.target

    data_train,data_test,target_train,target_test = train_test_split(Data,Target,test_size = 0.5)

    Classifier = tree.DecisionTreeClassifier()

    classifier = Classifier.fit(data_train,target_train)

    Prediction = Classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,Prediction)

    return Accuracy
    
def Accuracy_usingKNN():

    print("Welcome Case4: Accuracy using KNN algorithm")
    print(" ")

    Iris = load_iris()

    Data = Iris.data
    Target = Iris.target

    data_train,data_test,target_train,target_test = train_test_split(Data,Target,test_size = 0.5)

    Classifier = KNeighborsClassifier()

    classifier = Classifier.fit(data_train,target_train)

    Prediction = Classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,Prediction)

    return Accuracy


def main():
    print("_________________________________________")
    
    Ret = Accuracy_usingDecisionTree()
    print("Accurcy using decison tree algorithm: ",Ret*100)
    print(" ")
    Ret = Accuracy_usingKNN()
    print("Accuracy using KNN algorithm: ",Ret*100)

if __name__=="__main__":
    main()    
