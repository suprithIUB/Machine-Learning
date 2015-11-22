#Applied Machine Learning
#Programming Assignment -1
#Team Members:
#Suprith Chandrashekharachar
#Shreya Ghattamaraju Maruthi

import csv
import math
import pprint
from operator import getitem

'''To get class lavbel distribution of train set'''
def getClassLabelDistrbution(targetCol):
    for record in dataset:
        if record[targetCol] in classDist:
            classDist[record[targetCol]] += 1
        else:
            classDist[record[targetCol]] = 1

'''To get all class labels'''
def getClassLabels():
    return classDist.keys()

'''To get entropy of data set'''
def getDatasetEntropy():
    entropyOfDataset =0.00
    for classLabel in classDist:
        probabilityOfClass = classDist[classLabel]/float(totalNumberOfInstances)
        entropyOfDataset += -probabilityOfClass * (math.log(probabilityOfClass,2))
    return entropyOfDataset

'''To get all possible values for features from train set'''
def getValuesForFeature(dataset, attributeName):
    featureValues = list()
    for record in dataset:
        featureValues.append(record[attributeName])
    return featureValues

'''To get distribution of feature values'''
def getDistributionOfValues(listOfFeatureValues):
    valueDistributionMap = {}
    for value in listOfFeatureValues:
        if value in valueDistributionMap:
            valueDistributionMap[value] += 1
        else:
            valueDistributionMap[value] = 1
    return valueDistributionMap

'''To compute probability and log'''
def compute(value1, value2):
    prob = value1/float(value2)
    
    if prob > 0:
        return (prob * math.log(prob,2))
    else:
        return 0
    
'''To calculate data set entropy given different features '''
def getValueEntropy(value, setOfClasses, attribute, dataset, targetCol):
    instanceCount = 0.0
    totalEnt = 0.0
    totalCount = 0.0
    mapOfCounts = {}
    for record in dataset:
        if record[targetCol] in mapOfCounts:
            mapOfCounts[record[targetCol]] += 1
        else:
            mapOfCounts[record[targetCol]] = 1
    for eachKey in mapOfCounts:
        totalEnt -= compute(mapOfCounts[eachKey],float(sum(mapOfCounts.values())))
    return totalEnt
            
'''To get entropy'''    
def getEntropy(dataset,attribute, targetCol):
    listOfFeatureValues = getValuesForFeature(dataset, attribute)
    uniqueFeatureValues = set(listOfFeatureValues)
    setOfClasses = getClassLabels()
    
    valueDistributionMap = getDistributionOfValues(listOfFeatureValues)
    totalInstance = sum(valueDistributionMap.values())

    entropy = 0.0
    for value in valueDistributionMap:
        subsetOfRecords = []
        for record in dataset:
            if record[attribute] == value:
                subsetOfRecords.append(record)
                
        valueEntropy = getValueEntropy(value,setOfClasses, attribute, subsetOfRecords, targetCol)  
        temp = valueDistributionMap[value]/float(totalInstance) * valueEntropy
        entropy += temp
    
    return entropy

'''To get best feature while constructing decision tree'''
def getNextBestFeature(dataset, attributes, targetCol, removedAttr):
    bestGain = 0.0
    bestFeature =  0
    for attr in attributes:
        if attr not in removedAttr:
            ent = getEntropy(dataset, attr, targetCol)
            gain =  entropyOfDataset - ent
            if gain > bestGain:
                bestGain = gain
                bestFeature = attr
    return bestFeature

'''Creates a tree structure with node and its children'''        
class DecisionTreeNode(object):

    def __init__(self, leaf = None,feature = None, frequentClass = None):
        if leaf is None:
            self.leaf = "Nil"
        else:
            self.leaf = leaf
        if feature is None:
            self.feature = "Nil"
        else:
            self.feature = feature
        if frequentClass is None:
            self.frequentClass = "Nil"
        else:
            self.frequentClass = frequentClass
        self.children = {}

    def addChildNode(self, node):
        self.children.append(node)
        self.childrenCount += 1

    def getChildrenCount(self):
        return self.childrenCount
    
    def getChildren(self):
        return self.children
        
    def getLeftChild(self):
        return self.leftChild

    def getRightChild(self):
        return self.rightChild

    def getFeature(self):
        return self.feature

'''To split data set based on best feature'''
def getSubsetForFeatureValues(dataset, feature):

    mapOfSubsets = {}   
    for record in dataset:
            if record[feature] in mapOfSubsets:
                mapOfSubsets[record[feature]].append(record)
            else:
                mapOfSubsets[record[feature]] = [record]
    return mapOfSubsets
            
'''To get frequent class for a given feature'''
def getFrequentValue(dataset, targetAttribute):
    mapOfValues = {}
    freqVal = 0
    freqClass = 0
    for record in dataset:
        if record[targetAttribute] in mapOfValues:
            mapOfValues[record[targetAttribute]] += 1
        else:
            mapOfValues[record[targetAttribute]] = 1

    for val in mapOfValues:
        
        if mapOfValues[val] > freqVal:
            freqVal = mapOfValues[val]
            freqClass = val
    return freqClass


'''To create decision tree'''
def createDecisionTree(dataset, depth, targetAttribute, attributes, removedAttr = []):
    valuesForTargetAttr = getValuesForFeature(dataset, targetAttribute) #[record[targetAttribute] for record in dataset]
    frequentClass = getFrequentValue(dataset,targetAttribute)

    bF = getNextBestFeature(dataset, listOfAttr, targetAttribute, removedAttr)
    if not dataset or (len(attributes) - len(removedAttr)) <= 0:
        return DecisionTreeNode(True, "", frequentClass)
    elif valuesForTargetAttr.count(valuesForTargetAttr[0]) == len(valuesForTargetAttr):
        return DecisionTreeNode(True, valuesForTargetAttr[0], frequentClass)
    elif depth == depthOfTree:
        return DecisionTreeNode(False, bF, frequentClass)
    else:
        
        Node = DecisionTreeNode(False, bF, frequentClass)
        mapOfSubsets = getSubsetForFeatureValues(dataset, bF)
        removedAttr.append(bF)
        for value in mapOfSubsets:

            subtree = createDecisionTree(mapOfSubsets[value], depth+1, targetAttribute, attributes)
            Node.children[value] = subtree
            
    return Node

'''To traverse tree and get class labels'''
def getThePredcitionFromDecisionTree(recordInTestDataset, decisionTree):
    if decisionTree.leaf:
        return decisionTree.frequentClass
    else:
        children = decisionTree.children
        temp = recordInTestDataset[decisionTree.feature]
        if temp in children:
            return getThePredcitionFromDecisionTree(recordInTestDataset, children[temp])
        else:
            return decisionTree.frequentClass

'''To get confusion matrix for the test set'''
def confusionMatrix(mapItem , key):
    if key in mapItem:
        mapItem[key] += 1
    else:
        mapItem[key] = 1

'''To get class labels for test set'''
def predictClassLabel(testDataset, decisionTree, trueLabel):
    correctPredictions = 0
    wrongPredictions = 0
    for record in testDataset:
        
        classifiedLabel = getThePredcitionFromDecisionTree(record, decisionTree)
        if classifiedLabel == record[trueLabel]:
            confusionMatrix(truePositive , classifiedLabel)
            correctPredictions += 1
        else:
            confusionMatrix(error, (record[trueLabel], classifiedLabel))
            wrongPredictions += 1
            
    print "correctPredictions ... wrongPredictions"
    print correctPredictions,"...",wrongPredictions
    print ""
    return wrongPredictions

def printConfusionMatrix():
    print "Confusion Matrix"
    classLabels = getClassLabels()
    labels = len(classLabels)
    for i in range(1, labels+1):
        for j in range(1,labels+1):
            if i == j:
                if str(i) in truePositive:
                    print truePositive[str(i)],
                else:
                    print '0',
            else:
                key = (str(i),str(j))
                #print "key" , key
                if key in error:
                    print error[key],
                else:
                    print '0',
        print " "

trainSetPath = raw_input("Enter path of train set: ")
testSetPath = raw_input("Enter path of test set: ")
dataset = list( csv.reader( open( trainSetPath ) ) )
testDataset = list( csv.reader( open(testSetPath ) ) )

targetCol = eval(raw_input("Enter target Column: "))
depthOfTree = eval(raw_input("Enter depth of decision tree: "))
totalNumberOfInstances = len(dataset)

classDist = {}
getClassLabelDistrbution(targetCol)
entropyOfDataset = getDatasetEntropy()

listOfAttr = []
for i in range(0,len(dataset[0])-1):
    listOfAttr.append(i)

rootNode = createDecisionTree(dataset, 0, targetCol, listOfAttr)

truePositive = {}
error = {}

Misclassifications = predictClassLabel(testDataset, rootNode, targetCol)

print "Misclassifications " , Misclassifications
print ""
print "Accuracy" , (1 - (Misclassifications / float(len(testDataset)))) * 100, "%"
print ""

printConfusionMatrix()



