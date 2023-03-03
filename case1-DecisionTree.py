from sklearn import tree
# features = weight and surface
# labels = tennis and cricket

#Rough = 1
#smooth = 2

# cricket = 3
# tennis = 4

def MarvellousML(weight,surface):

    BallFeatures = [[35,1],[47,1],[90,2],[48,1],[90,2],[35,1],[92,2],[35,1],[35,1],[35,1],[96,2],[43,1],[110,2],[35,1],[95,2]]

    Names = [4,4,3,4,3,4,3,4,4,4,3,4,3,4,3]

    Cls = tree.DecisionTreeClassifier()
    Cls = Cls.fit(BallFeatures,Names)

    Result = Cls.predict([[weight,surface]])

    if Result == 3:
        print("Your object looks like: Cricket ball ")

    elif Result == 4:
        print("Your object looks like: Tennis ball")

def main():

    print("-------------------Case-1------------------")
    print(" ")

    print("Enter the weight")
    weight = int(input())

    print("Enter the surface rough or smooth")
    surface = input()

    if surface.lower()=="rough":
        surface = 1

    elif surface.lower()=="smooth":
        surface = 2

    else:
        print("ERROR : Wrong input")       

    MarvellousML(weight,surface)   

if __name__=="__main__":
    main()    

