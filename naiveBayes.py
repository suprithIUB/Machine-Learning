#Applied Machine Learning
#Programming Assignment - 2
#Team Members:
#Suprith Chandrashekharachar
#Shreya Ghattamaraju Maruthi
import csv

'''Get class label distribution for each class label'''
def getClassLabelDistribution(dataset, targetCol):
    classLabel = {}
    for record in dataset:
        if record[targetCol] not in classLabel:
            classLabel[record[targetCol]] = 1
        else:
            classLabel[record[targetCol]] += 1

    return classLabel

'''Applies laplace correction'''
def laplaceCorrection(numerator, denominator, discreteCount):

    prob = float(numerator +1) / (denominator + discreteCount)

    return prob

'''Get prior probability for each class label'''    
def getPriorProbability():
    classLabelCount = getClassLabelDistribution(dataset, targetCol)
    discreteValueCount = len(classLabelCount.keys())
    totalCount = len(dataset)
    

    priors = {}
    for item in classLabelCount:

        prob = laplaceCorrection(classLabelCount[item] , totalCount , discreteValueCount)
        priors[item] = prob

    return priors

'''Get values for a given feature'''
def getValuesForFeature(attributeName):
    featureValues = list()
    for record in dataset:
        featureValues.append(record[attributeName])
    return featureValues

'''Get individual conditional probability'''
def getIndividualConditionalProbability(feature, value, label, targetCol, possibleValueCount):
    countWithValueAndLabel = 0
    totalCountWithLabel = 0
    for record in dataset:
        if record[feature] == value and record[targetCol] == label:
            countWithValueAndLabel += 1
        if record[targetCol] == label:
            totalCountWithLabel += 1
    return laplaceCorrection(countWithValueAndLabel, totalCountWithLabel, possibleValueCount) 

'''Populate confusion matrix'''
def confusionMatrix(mapItem , key):
    if key in mapItem:
        mapItem[key] += 1
    else: mapItem[key] = 1

'''Print confusion matrix'''
def printConfusionMatrix():
    print "Confusion Matrix"
    classLabels = getClassLabelDistribution(dataset, targetCol)
    labels = len(classLabels.keys())
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

'''Naive Bayes Classifier
1) Trains on training set by counting values and performing Laplace correction.
2) Run on test set and generate models for both positive and negative class labels
3) Predict the label with higher probability.'''            
def naiveBayesClassifier(targetCol):
    priors = getPriorProbability()
    
    mapOfCondProbs = {}
    uniqueLabels = getClassLabelDistribution(dataset, targetCol).keys()
    for attribute in range(0, len(dataset[0])-1):
        
        valList = getValuesForFeature(attribute)
        if attribute == 12:
            valList.append('5')
        uniqueVals = set(valList)
        
        for val in uniqueVals:
            for label in uniqueLabels:
                mapOfCondProbs[str(attribute)+"_"+val+"_"+label] = getIndividualConditionalProbability(attribute, val, label, targetCol, len(uniqueVals))

    
    for record in test_dataset:
        positivePrediction = 1
        negativePrediction = 1
        bestLabel = None
        for attribute in range(0, len(record) -1):                
            positivePrediction *= mapOfCondProbs[str(attribute)+"_"+record[attribute]+"_"+str(1)]
            negativePrediction *= mapOfCondProbs[str(attribute)+"_"+record[attribute]+"_"+str(0)]
            

        positivePrediction = priors['1'] * positivePrediction
        negativePrediction = priors['0'] * negativePrediction

        positive = positivePrediction/(positivePrediction+negativePrediction)
        negative = negativePrediction/(negativePrediction+positivePrediction)
        
        predicted = None
        if positive > negative:
            predicted = 1
        else:
            predicted = 0
            
        trueLabel = int(record[targetCol])
        if predicted == trueLabel:
            confusionMatrix(truePositive, predicted)
        else:
            confusionMatrix(error,(trueLabel , predicted))
             
                                   
def misclassification():
    misclassifications = 0
    for item in error:
        misclassifications+=error[item]

    return misclassifications

truePositive = {}
error = {}
trainSetPath = raw_input("Enter path of train set: ")
testSetPath = raw_input("Enter path of test set: ")
dataset = list( csv.reader( open( trainSetPath) ))
test_dataset = list( csv.reader( open( testSetPath ) ))
targetCol = eval(raw_input("Enter target column: "))

naiveBayesClassifier(targetCol)

misclassifications = misclassification()
print "Misclassifications" , misclassifications
print "Accuracy" , ( 1 - (misclassifications / float(len(test_dataset)))) * 100, "%."
 
printConfusionMatrix()
