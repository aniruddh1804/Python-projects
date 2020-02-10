
def XTimesTable(x):
    return(lambda y : (y * x))


fourTimesTable = XTimesTable(4)
print(fourTimesTable(5))

myList = [1,2,3,4,5,6]

filterList = list(filter(lambda x : (x % 2 == 0), myList))
print(filterList)

mapList = list(map(lambda x : x * x, myList))
print(mapList)

# We can write all functions as lambdas
wholeSquare = lambda a,b : (a + b) ** 2
print(wholeSquare(5,3))



