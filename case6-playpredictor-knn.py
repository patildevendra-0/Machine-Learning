from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
from sklearn import preprocessing

def MarvellousPlayPredictor(Path):

    Data = pd.read_csv(Path,index_col = 0)

    print("Size of actual dataset: ",len(Data))

    feature_names = ['Whether','Temperature']
    print("Features of dataset: ",feature_names)

    whether = Data.Whether
    teampreture = Data.Temperature
    play = Data.Play 

    print("Whether :",whether)
    print("Tempreture: ",teampreture)
    print("play: ",play)

    le = preprocessing.LabelEncoder()

    whether_encoded = le.fit_transform(whether)
    print("Encoded whether: ",whether_encoded)

    teampreture_encoded = le.fit_transform(teampreture)
    print("Encoded tempreture: ",teampreture_encoded)

    label = le.fit_transform(play)
    Features = list(zip(whether_encoded,teampreture_encoded))

    model = KNeighborsClassifier()
    model.fit(Features,label)

    predicted = model.predict([[0,2]])  # 0 -overcast 2 mild

    predicted = le.inverse_transform(predicted)
    print(predicted)

       

def main():
    
    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__=="__main__":
    main()    