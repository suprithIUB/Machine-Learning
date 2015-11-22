import numpy as np
import sys
import copy

'''To create reward matrix with '-1' values'''
# def getRewardMatrx():
#     rewardMatrix = [[-1,-1,-1,10], \
#             [-1,-50,-1,-1], \
#             [-1,-1,-1,-1], \
#             [-1,-sys.maxint-1,-1,-sys.maxint-1], \
#             [-1,-1,-1,-1]]
#     return rewardMatrix

'''To create reward matrix with '0' values'''
def getRewardMatrx():
    rewardMatrix = [[0,0,0,10],\
                    [0,-50, 0, 0],\
                    [0,0,0,0],\
                    [0,-sys.maxint-1, 0, -sys.maxint-1],\
                    [0,0,0,0]]
    return rewardMatrix

'''To create the grid world for the agent'''
def createGrid():
    grid = [[0,0,0,0], \
            [0,0,0,0], \
            [0,0,0,0], \
            [0,0,0,0], \
            [0,0,0,0]]
    return grid

'''To check for legal actions in grid world domain
1. Don't allow agent to cross grid boundary
2. Don't allow agent to go into shaded regions/block'''
def checkBoundary(grid, state):
    rewardMatrix = getRewardMatrx()
    rows = len(grid)
    cols = len(grid[0])
    x = state[0]
    y = state[1]
    if x < 0 or x > rows-1:
        return False
    if y < 0 or y > cols-1:
        return False
    if rewardMatrix[x][y] == -sys.maxint-1:
        return False
    return True

'''To get the next moves from the current state'''
def getListOfNextStates():
    nextStates = {'up':(-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
    return nextStates

'''To get probabilities of all actions'''
def getTransitionProbability(nextStateString):
    transitionProbabilityMatrix = { 'left' : [1.0], \
                                    'down' : [1.0], \
                                    'right' : [0.8, 0.2], \
                                    'up' : [0.8, 0.2]}
    
    return transitionProbabilityMatrix[nextStateString]

'''To sum up values and transition porbabilities'''
def summation(probabilityList, valueList):
    val = 0
    for i in range(len(probabilityList)):
        val += probabilityList[i] * valueList[i]
    return val

'''To get the max value for every state by performing an action in grid world'''
def MaxValueFromBellmanEquation(gridWorld, gamma, state):
    nextStates = getListOfNextStates()
    rewardMatrix = getRewardMatrx()
    immediateReward = rewardMatrix[state[0]][state[1]]
    immediateValue = gridWorld[state[0]][state[1]]
    maxValue = -sys.maxint-1

    for action in nextStates.keys():
        coordinates =  tuple(sum(pair) for pair in zip (nextStates[action],state))
        valueList = list()
        probabilityList = getTransitionProbability(action)
        stateLegal = checkBoundary(gridWorld, coordinates)

        if not stateLegal: # bounce back to same state's value and reward
            valueList.append(immediateValue)
        else: # take value from the nex state
            valueList.append(gridWorld[coordinates[0]][coordinates[1]])

        if action == 'right':
            coordinates =  tuple(sum(pair) for pair in zip (nextStates['down'],state))
            stateLegal = checkBoundary(gridWorld, coordinates)
            if stateLegal:
                valueList.append(gridWorld[coordinates[0]][coordinates[1]])
            else:
                valueList.append(immediateValue)

        
        if action == 'up':
            coordinates =  tuple(sum(pair) for pair in zip (nextStates['left'],state))
            stateLegal = checkBoundary(gridWorld, coordinates)
            if stateLegal:
                valueList.append(gridWorld[coordinates[0]][coordinates[1]])
            else:
                valueList.append(immediateValue)

        value = immediateReward + gamma * summation(probabilityList, valueList)
        maxValue = max(value, maxValue)
    return maxValue

'''To check for convergence of values'''
def converged(previous, current):
    mat1 = np.matrix(previous)
    mat2 = np.matrix(current)
    if np.allclose(mat2, mat1):
        return True
    else:
        return False

'''To print values of each state'''
def printQValues(gridWorld):

    print "-" *100
    for row in gridWorld:

        for item in row:
            print "|" , '%15s' %str(item) , "\t",

        print "|"
        print "-" * 100

'''To perform value iteration over all states in the grid world domain'''
def valueIteration():
    gamma = 0.9
    gridWorld = createGrid()

    previous = copy.deepcopy(gridWorld)
    RewardMatrx = getRewardMatrx()
    count=0
    while(True):    
        for row in range(len(gridWorld)):
            for column in range(len(gridWorld[0])):
                if RewardMatrx[row][column] != -sys.maxint-1:
                    gridWorld[row][column] = MaxValueFromBellmanEquation(gridWorld, gamma, (row, column))                    
        
        if(converged(previous, gridWorld)):
            break
        else:
            previous = copy.deepcopy(gridWorld)
        count += 1

    printQValues(gridWorld)
        
valueIteration()

