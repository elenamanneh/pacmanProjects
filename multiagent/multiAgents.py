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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = successorGameState.getScore()

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 1.0 / min(foodDistances)

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            
            if scaredTime > 0:
                score += 10.0 / (ghostDistance + 1)
            else:
                if ghostDistance <= 1:
                    score -= 200 # some large penalty

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max(minimax((agentIndex + 1) % gameState.getNumAgents(), depth + (agentIndex + 1) // gameState.getNumAgents(), gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
            else: 
                return min(minimax((agentIndex + 1) % gameState.getNumAgents(), depth + (agentIndex + 1) // gameState.getNumAgents(), gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))

        pacmanActions = gameState.getLegalActions(0) 
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in pacmanActions]
        bestScore = max(scores)
        bestIndices = [index for index, score in enumerate(scores) if score == bestScore]
        chosenIndex = bestIndices[0]
        return pacmanActions[chosenIndex]
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            numAgents = gameState.getNumAgents()
            if agentIndex == 0: 
                v = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, alphaBeta((agentIndex + 1) % numAgents, depth + (agentIndex + 1) // numAgents, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if v > beta:
                        return v 
                    alpha = max(alpha, v)
                return v
            else: 
                v = float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, alphaBeta((agentIndex + 1) % numAgents, depth + (agentIndex + 1) // numAgents, gameState.generateSuccessor(agentIndex, action), alpha, beta))
                    if v < alpha:
                        return v 
                    beta = min(beta, v)
                return v

        alpha = float("-inf")
        beta = float("inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0): 
            score = alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if score > alpha:
                alpha = score
                bestAction = action
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max(expectimax((agentIndex + 1) % gameState.getNumAgents(), depth + (agentIndex + 1) // gameState.getNumAgents(), gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
            else: 
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + (agentIndex + 1) // gameState.getNumAgents()
                actions = gameState.getLegalActions(agentIndex)
                return sum(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in actions) / len(actions)

        pacmanActions = gameState.getLegalActions(0) 
        scores = [expectimax(1, 0, gameState.generateSuccessor(0, action)) for action in pacmanActions]
        bestScore = max(scores)
        bestIndices = [index for index, score in enumerate(scores) if score == bestScore]
        chosenIndex = bestIndices[0]
        return pacmanActions[chosenIndex]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The code defines an evaluation function that calculates a
    score for each state to guide Pacman's decisions. It calculates the 
    score factoring in the distance to the nearest food, to encourage
    food collection. It also considers ghost proximity, penalizing states 
    where ghosts are close and rewarding states where scared ghosts are 
    close. It also deducts points for remaining power capsules, to 
    encourage their consumption, and for remaining food, to incentivize
    clearing the level efficiently.
    """
    "*** YOUR CODE HERE ***"
    
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    foodDistances = [util.manhattanDistance(pacmanPosition, foodPos) for foodPos in food.asList()]
    if foodDistances:
        score += 1.0 / min(foodDistances)

    for ghostState in ghostStates:
        ghostDistance = util.manhattanDistance(pacmanPosition, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            score += 10.0 / (ghostDistance + 1)
        else:
            if ghostDistance <= 2:
                score -= 200 / (ghostDistance + 1)

    score -= 20 * len(capsules)
    score -= 4 * len(foodDistances)

    return score

# Abbreviation
better = betterEvaluationFunction
