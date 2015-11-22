author__ = 'shreya'


import sys
import random

'''To create grid to hold values for all actions for every state'''
def createGrid():
    grid = [[(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)], \
            [(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)], \
            [(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)], \
            [(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)], \
            [(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)]]
    return grid

'''To create reward matrix with '-1' values'''
# def getRewardMatrix():
#     rewardMatrix = [[-1,-1,-1,10], \
#             [-1,-50,-1,-1], \
#             [-1,-1,-1,-1], \
#             [-1,-sys.maxint-1,-1,-sys.maxint-1], \
#             [-1,-1,-1,-1]]
#     return rewardMatrix

'''To create reward matrix with '0' values'''
def getRewardMatrix():

    rewardMatrix = [[0,0,0,10],\
                    [0,-50, 0, 0],\
                    [0,0,0,0],\
                    [0,-sys.maxint-1, 0, -sys.maxint-1],\
                    [0,0,0,0]]

    return rewardMatrix

'''To check for legal actions in grid world domain
1. Don't allow agent to cross grid boundary
2. Don't allow agent to go into shaded regions/block'''
def checkBoundary(grid, state):

    rewardMatrix = getRewardMatrix()
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

'''Maps to keep track of actions'''
'''(up, down, left, right)'''
actionMap = {'up': 0 , 'down':1 , 'left':2, 'right':3}
reverseMap = {0:'up' , 1:'down' , 2:'left' , 3:'right'}

'''To find out which action was performed'''
def getActionPerformed(moveMade):
    return actionMap[moveMade]

'''To select randomly if agent should explore or exploit'''
def exploreExpoilt(epsilon):

    choice = random.uniform(0 , 1)

    if choice < epsilon:
        return 0#explore
    else:
        return 1#exploit

'''If explore was chosen, then pick a random action to perform'''
def exploitStates(qValues):

    if max(qValues) == 0:
        qValueIndex = random.randrange(len(qValues))
        return reverseMap[qValueIndex]
    else:
        qValueIndex = qValues.index(max(qValues))
        return reverseMap[qValueIndex]

'''To calculate new Q value for a particular action in a particular state'''
def computeNewQValue(gridWorld , state, reward, action, nextPosition, alpha, gamma):

    maxQ = max(gridWorld[nextPosition[0]] [nextPosition[1]])
    currentQ = gridWorld[state[0]] [state[1]][action]
    noiseLevel = reward + (gamma * maxQ) - currentQ
    qValue = round(currentQ + (alpha * noiseLevel), 4)

    values = list(gridWorld[state[0]][state[1]])
    values[action] = qValue

    return values

'''To make the agent choose its next action between explore and exploit.
 And to pick an action and keep updating q-values till agent reaches goal'''
def getQValueForStates(gridWorld, gamma, alpha, epsilon, state):

    nextStates = getListOfNextStates()
    rewardMatrix = getRewardMatrix()

    currentQValue = gridWorld[state[0]] [state[1]]
    currentReward = rewardMatrix[state[0]] [state[1]]

    # Check if agent reached goal
    if state == (0, 3):
        if gridWorld [state[0]]  [state[1]] == (0,0,0,0):
            goalQValue = (10, 0, 0, 10)
            gridWorld [state[0]]  [state[1]] = goalQValue

        else:

            actionToUpdate = actionMap.values()
            for action in actionToUpdate:
                values = computeNewQValue(gridWorld, state, currentReward, action, state, alpha, gamma)
                gridWorld[state[0]][state[1]] = tuple(values)
        return (-1,-1)

    else:

        typeOfRL = exploreExpoilt(epsilon)
        if typeOfRL == 0:

            move = random.choice(nextStates.keys())
            nextPosition =  tuple(sum(pair) for pair in zip (nextStates[move],state))
            isStateLegal = checkBoundary(gridWorld, nextPosition)
            action = getActionPerformed(move)

            if not isStateLegal:
                values = computeNewQValue(gridWorld, state, currentReward, action, state, alpha, gamma)
                gridWorld[state[0]][state[1]] = tuple(values)
                return state

        else:

            move = exploitStates(currentQValue)
            nextPosition =  tuple(sum(pair) for pair in zip (nextStates[move],state))
            isStateLegal = checkBoundary(gridWorld, nextPosition)
            action = getActionPerformed(move)

            if not isStateLegal:
                values = computeNewQValue(gridWorld, state, currentReward, action, state, alpha, gamma)
                gridWorld[state[0]][state[1]] = tuple(values)
                return state

    values = computeNewQValue(gridWorld, state, currentReward, action, nextPosition, alpha, gamma)
    gridWorld[state[0]][state[1]] = tuple(values)

    return nextPosition


'''To print values of each state and each action'''
def printQValues(gridWorld):

    print "-" *130
    for row in gridWorld:

        for item in row:
            print "|" ,"\t  ", '%12s' %str(item[0]) ,"\t  ",
        print ""

        for item in row:
            print "|" , '%8s' %str(item[2]),"\t  ",  '%15s' %str(item[3]),
        print ""

        for item in row:
            print "|" ,"\t  ", '%12s' %str(item[1]) ,"\t  ",
        print ""

        print "-" *130

'''To run q-learning for 5000 episodes'''
def qLearning():
    epsilon = 0.5
    gamma = 0.9
    alpha = 0.1
    gridWorld = createGrid()

    level = 0
    for i in range(0, 5000):

        level+=1

        if level == 10:
            epsilon = (1-epsilon) / (2-epsilon)
            level = 0

        position = (4,0)
        while position != (-1,-1):
            position = getQValueForStates(gridWorld, gamma, alpha, epsilon, position)

    printQValues(gridWorld)

qLearning()
