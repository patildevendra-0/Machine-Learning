import numpy as np 
from sklearn import tree
from sklearn.datasets import load_iris

Iris = load_iris()

print("Features from iris dataset: ")
print(Iris.feature_names)

print("Target from iris dataset: ")
print(Iris.target_names)

# index of removed elements for testing
test_index = [1,51,101]

#tarining data with removed elements
train_data = np.delete(Iris.data,test_index,axis = 0)
train_target = np.delete(Iris.target,test_index)

#testing data for tes the training data
testing_data = Iris.data[test_index]
testing_target = Iris.target[test_index]

#from classifier decision tree classifier
classifier = tree.DecisionTreeClassifier()

#apply trainig data to form tree
classifier.fit(train_data,train_target)

print("testing target is")
print(testing_target)

#test tha model
print("Result is :")
print(classifier.predict(testing_data))