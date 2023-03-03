from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def euc(a,b):
    Dis = distance.euclidean(a,b)
    return Dis

class MarvellousUserKNN():

    def fit(self,TrainingData,TrainingTarget):
        self.TrainingData = TrainingData
        self.TrainingTarget = TrainingTarget

    def predict(self,TestingData):

        Prediction = []

        for row in TestingData:
            label = self.closest(row)
            Prediction.append(label)
        print("predicted",Prediction)    
        return Prediction

    def closest(self,row):
        bestdistnce = euc(row,self.TrainingData[0])
        bestindex = 0

        for i in range(len(self.TrainingData)):
            dist = euc(row,self.TrainingData[i])

            if dist<bestdistnce:
                bestdistnce = dist
                bestindex = i
        return self.TrainingTarget[bestindex]

def MarvellousKNN():

    Iris = load_iris()

    Data = Iris.data
    Target = Iris.target

    print("__"*50)
    print(" ")
    print("Features of data set: ",Iris.feature_names)
    print("__"*50)
    print("Labels or target of dataset: ",Iris.target_names)
    print("__"*50)

    print("Dataset before splitting and shuffle: ")
    for i in range(len(Iris.target)):
        print("ID :%d  Labels :%s  Features :%s"%(i,Data[i],Target[i]))
    print("Actual Dataset : ",(i+1))    
    print("__"*50)    

    data_train,data_test,target_train,target_test = train_test_split(Data,Target,test_size=0.5)

    print("__"*50)
    print("Trainig Dataset:  ")
    for i in range(len(data_train)):
        print("ID :%d  Labels :%s  Features :%s"%(i,data_train[i],target_train[i]))
    print("Actual Dataset : ",(i+1))
    print("__"*50)

    print("__"*50)
    print("Testing Dataset:  ")
    for i in range(len(data_test)):
        print("ID :%d  Labels :%s  Features :%s"%(i,data_test[i],target_test[i]))
    print("Actual Dataset : ",(i+1))
    print("__"*50)

    Classifier = MarvellousUserKNN()
    Classifier.fit(data_train,target_train)
    predicted = Classifier.predict(data_test)
    Accuracy = accuracy_score(target_test,predicted)

    return Accuracy  

def main():
    accu = MarvellousKNN()
    print("Accuracy of userdefined KNN is: ",accu*100)

if __name__=="__main__":
    main()                       
