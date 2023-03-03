
from sklearn.datasets import load_iris

Iris = load_iris()

print("Features of Iris data set :")
print(Iris.feature_names)

print(" ")
print("Target/labels of Iris data set")
print(Iris.target_names)

print("Elements of Data set : ")

for i in range(len(Iris.target)):
    print("ID: %d ,Feature: %s ,Lables: %s"%(i,Iris.data[i],Iris.target[i]))