# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        """
        If new state contains a ghost, then it is a very bad state to move for pacman 
        so score returned is - infinity. 
        """
        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos:
                return float("-inf")

        currFood = currentGameState.getFood().asList()

        """
        We will return more score for that state which contains food at minimum min_distance
        from current state. More will be the min_distance, less will be the score so returning
        inverse of distance as the factor influencing score of the state.
        """
        min_distance = float("inf")
        for food in currFood:
            if Directions.STOP in action:
                return float("-inf")
            foodDistance_FromNewPos = manhattanDistance(food, newPos)
            if foodDistance_FromNewPos < min_distance:
                min_distance = foodDistance_FromNewPos

        return  1000.0 / (1000.0 + min_distance)

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Initializing current depth to 0
        currDepth = 0
        optimalValAction = self.getValue(gameState,self.index,currDepth)
        nextOptimalMove = optimalValAction[1]
        return nextOptimalMove
    
    def isPacman(self, index):
        return index == 0
    
    #Function to switch between pacman and ghost states which are essentially
    # min-max states with multiple min levels. The function is caller recursively 
    # in order to have a DFS implementation for the search.
    def getValue(self,gameState,agentIndex,currDepth):     
        
        # Depth should be increased by 1 when all the agents have played their chance once.
        if agentIndex == gameState.getNumAgents() :
            agentIndex = 0
            currDepth = currDepth + 1 

        legalActions = []
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Reached terminal state as there are no legal action to perform, return with the value calculated 
        # by evaluation function
        if len(legalActions) == 0:
            return [self.evaluationFunction(gameState),None]
        
        # Explored till specified depth, should terminate search here.
        if currDepth == self.depth:
            return [self.evaluationFunction(gameState),None]
        
        # If its Max node( or PacMan) then call the maxValue function else call the minValue function 
        # for all the adversarieal agents.
        
        
        if self.isPacman(agentIndex):
            return self.maxValue(gameState, agentIndex, currDepth)
        else:
            return self.minValue(gameState, agentIndex, currDepth)
            
            
    def minValue(self, gameState, agentIndex, depth):
        value = float("inf")
        currValAction = ()        
        
        legalActions = []
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Call recursively for each possible action and propogate the value in
        # bottom up fashion from the terminal nodes. Min node always takes 
        # the minimum value of its children(MAX node).
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action) 
            val, a = self.getValue(succ, agentIndex + 1, depth)
            
            if val < value:
                value = val
                currValAction = (value,action)
        
        return currValAction      
    
    def maxValue(self, gameState, agentIndex, depth):
        value = -float("inf")
        currValAction = ()
        
        legalActions = []
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Call recursively for each possible action and propogate the value in
        # bottom up fashion from the terminal nodes. Max node always takes
        # the maximum value of the children(MIN node)       
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action) 
            val,a = self.getValue(succ, agentIndex + 1, depth)
            
            if val > value:
                value = val
                currValAction = (value,action)
        
        return currValAction
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Setting current depth to zero
        # alpha value to -infinity
        # beta value to +infinity
        
        currDepth = 0
        alphaVal = -float("inf")
        betaVal = float("inf")
        optimalValAction = self.getValue(gameState,self.index,currDepth, alphaVal, betaVal)
        nextOptimalMove = optimalValAction[1]
        return nextOptimalMove
    
    def isPacman(self, index):
        return index == 0
    
    def getValue(self,gameState,agentIndex,currDepth, alphaVal, betaVal):     
        # Depth should be increased by 1 when all the agents have played their chance once.
        if agentIndex == gameState.getNumAgents() :
            agentIndex = 0
            currDepth = currDepth + 1 

        legalActions = []
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Reached terminal state as there are no legal action to perform, return with the value calculated 
        # by evaluation function
        if len(legalActions) == 0:
            return (self.evaluationFunction(gameState),None)
        
        # Explored till specified depth, should terminate search here.
        if currDepth == self.depth:
            return (self.evaluationFunction(gameState),None)
        
        # If its Max node( or PacMan) then call the maxValue function else call the minValue function 
        # for all the adversarieal agents.
        
        if self.isPacman(agentIndex):
            return self.maxValue(gameState, agentIndex, currDepth, alphaVal, betaVal)
        else:
            return self.minValue(gameState, agentIndex, currDepth ,alphaVal, betaVal)
            
            
    def minValue(self, gameState, agentIndex, depth, alphaVal, betaVal):
        value = float("inf")
        currValAction = ()
        legalActions = gameState.getLegalActions(agentIndex)
        
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action) 
            val,a = self.getValue(succ, agentIndex + 1, depth,alphaVal, betaVal)
            
            if val < value:
                value = val
                currValAction = (value,action)
           
            # If the value returned is less than alpha value then we should prune/cut our
            # search here as no matter how much we search ahead in this subtree max node already
            # has a better value from some other path and will take that anyhow.
            if value < alphaVal:
                return currValAction
            
            # Update beta value for the min nodes so that further search can prevent unnecessary 
            # work of looking for a better solution which it will not find.
            betaVal = min(betaVal, value)
        
        return currValAction      
    
    def maxValue(self, gameState, agentIndex, depth, alphaVal, betaVal):
        value = -float("inf")
        currValAction = ()
        
        legalActions = []
        legalActions = gameState.getLegalActions(agentIndex)
        
        for action in legalActions:
            succ = gameState.generateSuccessor(agentIndex, action)
            val, a = self.getValue(succ, agentIndex + 1, depth, alphaVal, betaVal)
            
            if val > value:
                value = val
                currValAction = (value,action)
                
            # If the value returned is greater value beta then we should prune/cut our
            # search here as to no matter how much we search ahead in this subtree min node already
            # has a better value from some other path in this subtree and will always take that. 
            # Also MAX node will be taking the other option which returned a greater value.
            if value > betaVal:
                return currValAction
            
            alphaVal = max(alphaVal, value)
            
        return currValAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maximumVal(gameState, self.index, self.depth)[1]

    def maximumVal(self, gameState, indexOfAgent, depth):

        # if game is lost already
        if gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        # if game is win already
        if gameState.isWin():
            return self.evaluationFunction(gameState), 'Stop'

        #if game is at depth == 0
        if depth == 0:
            return self.evaluationFunction(gameState), 'Stop' 

        successorStates = []
        actions = gameState.getLegalActions(indexOfAgent)

        for action in actions:
            successorStates.append(gameState.generateSuccessor(indexOfAgent, action))

        scoreVals = []
        for successorState in successorStates:
            scoreVals.append(self.expectedVal(successorState, indexOfAgent + 1, depth)[0])

        bestPossibleScore = max(scoreVals)

        length = len(scoreVals)
        bestIndices = []
        for i in range(length):
            if scoreVals[i] == bestPossibleScore:
                bestIndices.append(i)
          
        randomIndex = random.choice(bestIndices)

        return bestPossibleScore, actions[randomIndex]

    def expectedVal(self, gameState, indexOfAgent, depth):

        # if game is lost already
        if gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        # if game is win already
        if gameState.isWin():
            return self.evaluationFunction(gameState), 'Stop'

        #if game is at depth == 0
        if depth == 0:
            return self.evaluationFunction(gameState), 'Stop' 

        successorStates = []
        actions = gameState.getLegalActions(indexOfAgent)

        for action in actions:
            successorStates.append(gameState.generateSuccessor(indexOfAgent, action))

        scoreVals = []
        # if there is next ghost, then call expectedVal, else call maximumVal function
        if indexOfAgent < gameState.getNumAgents() - 1:
            for successorState in successorStates:
                scoreVals.append(self.expectedVal(successorState, indexOfAgent + 1, depth)[0])
        else:
            for successorState in successorStates:
                scoreVals.append(self.maximumVal(successorState, 0, depth - 1)[0])

        # Returning the average value
        s = sum(scoreVals)
        length = len(scoreVals)
        bestPossibleScore = float(s) / length

        return bestPossibleScore, 'Stop'

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

