#Applied Machine Learning
#Programming Assignment - 2
#Team Members:
#Suprith Chandrashekharachar
#Shreya Ghattamaraju Maruthi

import csv
import math
import copy

'''To get distribution of class labels'''
def getClassLabel(dataset, targetCol):
    classLabel = []
    for record in dataset:
        if record[targetCol] not in classLabel:
            classLabel.append(record[targetCol])

    return classLabel

'''To separate data points and class label'''
def getDataPoints(dataRow):
    data = [int(num) for num in dataRow[0:16]]
    trueLabel = int (dataRow[targetCol])
    return data , trueLabel

'''To get discrete values for feature with discrete values'''
def getDiscreteValues(attribute):
    dicreteValues = {}
    for record in dataset:
        if record[attribute] in dicreteValues:
            dicreteValues[record[attribute]]+=1
        else:
            dicreteValues[record[attribute]] = 1

    return dicreteValues

'''To get possible feature values for discrete values feature'''
def getFeatureValues():
    oneVsRestFeature = []
    for i in range(0,16):
        dicreteValues = getDiscreteValues(i)
        if len(dicreteValues.keys()) > 2:
            oneVsRestFeature.append(i)
            
    return oneVsRestFeature

'''To make 1 vs rest model for each discrete valued feature'''
def makeNewModel(feature, value , numberOfWeights):  
    dataset1 = copy.deepcopy(dataset)
    for record in dataset1:
        if int(record[feature]) == int(value):
            record[feature] = 1
        else:
            record[feature] = 0

    return dataset1

'''To create the dataset for 1 vs rest model'''    
def createOnevsRest():
    oneVsRestFeature = getFeatureValues()
    weights = {}

    for feature in oneVsRestFeature:

        discreteValues = getDiscreteValues(feature)
        possibleValues = discreteValues.keys()

        for item in discreteValues.keys():
            dataset1 = makeNewModel(feature, item , possibleValues)

            weightVector = logisticRegression(dataset1, targetCol)
            weights[(feature,item)] = weightVector
            
    return weights

'''Perform logistic regression on the train set.
Checks for convergence. If difference between all corresponding weight elements
between previous and current weight is less than learning rate value.
Then values have converged.'''    
def logisticRegression(trainingData, targetCol):   
    weightVector = [0] * targetCol    
    level = 0
    while True:

        level+=1
        gradient = [0] * targetCol

        for record in trainingData:

            data, trueLabel = getDataPoints(record)
            wxProduct = sum([a*b for (a,b) in zip(weightVector , data)])
            eWX = math.exp(-wxProduct)
            probability = 1 / ( 1 + eWX)
            error = trueLabel - probability

            for j in range(0 , targetCol):
                gradient[j] = gradient[j] + error * data[j]

            prevWeight = copy.copy(weightVector)

            for k in range(0 , targetCol):
                weightVector[k] = weightVector[k] + eta * gradient[k]

            currWeight = copy.copy(weightVector)

        misCount = 0
        for i in range(len(prevWeight)):
            if (prevWeight[i] - currWeight[i]) > eta:
                misCount+=1
                
        if misCount == 0:
            break

    return weightVector

'''To get all test points and process them one at a time.'''
def getAllTestPoints(dataRow, weights):   
    weightKeys = weights.keys()
    predictions = []
    data, trueLabel = getDataPoints(dataRow)
    foundValue = 0
    
    for item in weightKeys:
        attribute = item[0]
        value = item[1]
        foundValue = value
        found = False
        
        if int(dataRow[attribute]) == int(value):
            
            found = True
            dataRow[attribute] = '1'
            weightVector = weights[(attribute , str(value))]
            pred = getPredictions(dataRow, weightVector)
            predictions.append(pred)
            break

    if found:
        for item in weightKeys:
            attribute = item[0]
            value = item[1]

            if int(dataRow[attribute]) != foundValue:
                dataRow[attribute] = '0'
                
            weightVector = weights[(attribute , str(value))]
            pred = getPredictions(dataRow, weightVector)
            predictions.append(pred)
    else:
        for item in weightKeys:
            
            attribute = item[0]
            value = item[1]
            dataRow[attribute] = '0'
            weightVector = weights[(attribute , str(value))]
            pred = getPredictions(dataRow, weightVector)
            predictions.append(pred)

    predicted = getAveragePrediction(predictions)

    if predicted > 0:
        predictedLabel = 1
    else:
        predictedLabel = 0
        
    if predictedLabel == trueLabel:
        confusionMatrix(truePositive , predictedLabel)

    else:
        confusionMatrix(error, (trueLabel , predictedLabel))

'''To get average of prediction because there are many weight vectors'''
def getAveragePrediction(listItem):    
    sumItem = 0
    for item in listItem:
        sumItem+= item
    prediction = sumItem / float(len(listItem))
    return prediction

'''To perform "wx" dot product and classify test set.'''    
def getPredictions(dataRow, weightVector):    
    data, trueLabel = getDataPoints(dataRow)
    pred = 0
    for i in range(len(weightVector)):
        pred+= (weightVector[i] * data[i])
    return pred

'''To classify all test points'''    
def classifyTestset(weights):
    for record in testset:
        getAllTestPoints(record, weights)

'''To create confusion matrix'''
def confusionMatrix(mapItem , key):
    if key in mapItem:
        mapItem[key] += 1
    else:
        mapItem[key] = 1

'''To print confusion matrix'''
def printConfusionMatrix():
    print "Confusion Matrix"
    classLabels = getClassLabel(dataset, targetCol)
    labels = len(classLabels)
    for i in range(0, labels):
        for j in range(0,labels):
            if i == j:
                if i in truePositive:
                    print truePositive[i],
                else:
                    print '0',
            else:
                key = (i,j)
                if key in error:
                    print error[key],
                else:
                    print '0',
        print " "    

'''To get number of misclassifications'''
def misclassification():
    misclassifications = 0
    for item in error:
        misclassifications+=error[item]

    return misclassifications

trainSetPath = raw_input("Enter path of train set: ")
testSetPath = raw_input("Enter path of test set: ")

dataset = list( csv.reader( open( trainSetPath) ))
targetCol = eval(raw_input("Enter target column: "))
eta = eval(raw_input("Enter learning rate: "))
weights = createOnevsRest()

testset = list( csv.reader( open( testSetPath ) ))
truePositive = {}
error = {}
classifyTestset(weights)
printConfusionMatrix()
misclassifications = misclassification()
print "Misclassifications" , misclassifications
print "Accuracy" , ( 1 - (misclassifications / float(len(testset)))) * 100, "%."
    
    
