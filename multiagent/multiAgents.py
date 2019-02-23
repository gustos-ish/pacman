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

import random
import numpy as np
import multiagent.util as util

from multiagent.game import Agent
from multiagent.util import manhattanDistance, matrixAsList


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
        legal_moves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

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
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        foods = currentGameState.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]

        "*** YOUR CODE HERE ***"
        ##########################
        # SCARED TIMES
        ##########################
        score_scared_times = sum(new_scared_times)

        ##########################
        # GHOST DISTANCE
        ##########################
        minus_inf = - 10 ** 10
        new_ghosts_positions = [ghost_state.configuration.pos for ghost_state in new_ghost_states]
        dist_new_pos_ghosts = [manhattanDistance(xy1=new_pos, xy2=ghost) for ghost in new_ghosts_positions]
        nearest_ghost_dist = min(dist_new_pos_ghosts)
        score_ghost_dist = - 2 / nearest_ghost_dist if nearest_ghost_dist != 0 else minus_inf
        if nearest_ghost_dist == 0:
            return score_ghost_dist

        ##########################
        # FOOD SCORE
        ##########################
        foods_positions = matrixAsList(matrix=foods.data, value=True)
        nearest_food = min([manhattanDistance(xy1=new_pos, xy2=food) for food in foods_positions])
        score_food = 1 / nearest_food if nearest_food != 0 else 2

        ##########################
        # TOTAL SCORE
        ##########################
        if score_scared_times > 0:
            score = score_food
        else:
            score = score_ghost_dist + score_food

        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        best_action = self.minimax_decision(game_state=gameState, depth=0)
        return best_action

    def minimax_decision(self, game_state, depth):
        return self.max_agent_decision(game_state=game_state, depth=depth)

    def max_agent_decision(self, game_state, depth):
        """
        Returns the action which maximizes the score
        after the next depth rounds of the game

        :param game_state:
        :param depth:
        :return:
        """
        if game_state.isWin() or game_state.isLose():
            return game_state.getScore()

        pacman_index = 0
        best_action = None
        max_score = - np.inf
        pacman_actions = game_state.getLegalActions(agentIndex=pacman_index)
        for action in pacman_actions:
            first_ghost_index = 1
            score = self.min_agent_score(
                game_state=game_state.generateSuccessor(agentIndex=pacman_index, action=action),
                depth=depth,
                ghost_index=first_ghost_index
            )
            if score > max_score:
                max_score = score
                best_action = action

        if depth == 0:
            return best_action

        return max_score

    def min_agent_score(self, game_state, depth, ghost_index):
        """
        Returns the minimal score
        after the next depth rounds of the game

        :param game_state:
        :param depth:
        :param ghost_index:
        :return:
        """
        if game_state.isLose() or game_state.isWin():
            return game_state.getScore()

        pacman_index = 0
        min_score = np.inf
        next_agent_index = (ghost_index + 1) % game_state.getNumAgents()
        actions = game_state.getLegalActions(agentIndex=ghost_index)
        for action in actions:
            if next_agent_index == pacman_index:
                if depth == self.depth - 1:
                    score = self.evaluationFunction(game_state.generateSuccessor(agentIndex=ghost_index, action=action))
                else:
                    score = self.max_agent_decision(
                        game_state=game_state.generateSuccessor(agentIndex=ghost_index, action=action),
                        depth=depth + 1
                    )
            else:
                score = self.min_agent_score(
                    game_state=game_state.generateSuccessor(agentIndex=ghost_index, action=action),
                    depth=depth,
                    ghost_index=next_agent_index
                )
            min_score = min(score, min_score)

        return min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_action = self.alpha_beta_minimax_decision(game_state=gameState, depth=0)
        return best_action

    def alpha_beta_minimax_decision(self, game_state, depth):
        alpha_start = - np.inf
        beta_start = np.inf
        return self.max_agent_decision(game_state=game_state, depth=depth, alpha=alpha_start, beta=beta_start)

    def max_agent_decision(self, game_state, depth, alpha, beta):
        """

        :param game_state:
        :param depth:
        :param alpha:
        :param beta:
        :return:
        """
        if game_state.isLose() or game_state.isWin():
            return game_state.getScore()

        pacman_index = 0
        best_action = None
        max_score = - np.inf
        actions = game_state.getLegalActions(agentIndex=pacman_index)
        for action in actions:
            first_ghost_index = 1
            action_score = self.min_agent_score(
                game_state=game_state.generateSuccessor(agentIndex=pacman_index, action=action),
                depth=depth,
                ghost_index=first_ghost_index,
                alpha=alpha,
                beta=beta,
            )
            if action_score > max_score:
                max_score = action_score
                best_action = action

            if max_score > beta:
                return max_score

            alpha = max(alpha, max_score)

        if depth == 0:
            return best_action

        return max_score

    def min_agent_score(self, game_state, depth, ghost_index, alpha, beta):
        """

        :param game_state:
        :param depth:
        :param ghost_index:
        :param alpha:
        :param beta:
        :return:
        """
        if game_state.isLose() or game_state.isWin():
            return game_state.getScore()

        pacman_index = 0
        min_score = np.inf
        actions = game_state.getLegalActions(agentIndex=ghost_index)
        next_agent_index = (ghost_index + 1) % game_state.getNumAgents()
        for action in actions:
            if next_agent_index == pacman_index:
                if depth == self.depth - 1:
                    score = self.evaluationFunction(game_state.generateSuccessor(agentIndex=ghost_index, action=action))
                else:
                    score = self.max_agent_decision(
                        game_state=game_state.generateSuccessor(agentIndex=ghost_index, action=action),
                        depth=depth + 1,
                        alpha=alpha,
                        beta=beta,
                    )
            else:
                score = self.min_agent_score(
                    game_state=game_state.generateSuccessor(agentIndex=ghost_index, action=action),
                    depth=depth,
                    ghost_index=next_agent_index,
                    alpha=alpha,
                    beta=beta,
                )

            min_score = min(score, min_score)
            if min_score < alpha:
                return min_score

            beta = min(beta, min_score)

        return min_score


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
        best_action = self.expectimax_decision(game_state=gameState, depth=0)
        return best_action

    def expectimax_decision(self, game_state, depth):
        return self.max_agent_decision(game_state=game_state, depth=depth)

    def max_agent_decision(self, game_state, depth):
        """

        :param game_state:
        :param depth:
        :return:
        """
        if game_state.isLose() or game_state.isWin():
            return game_state.getScore()

        pacman_index = 0
        best_action = None
        max_score = - np.inf
        actions = sorted(game_state.getLegalActions(agentIndex=pacman_index))
        #if depth == 0:
        #    print(actions)

        for action in actions:
            first_ghost_index = 1
            action_score = self.min_agent_score(
                game_state=game_state.generateSuccessor(agentIndex=pacman_index, action=action),
                depth=depth,
                ghost_index=first_ghost_index,
            )

            if action_score > max_score:
                max_score = action_score
                best_action = action

        if depth == 0:
            return best_action

        return max_score

    def min_agent_score(self, game_state, depth, ghost_index):
        """

        :param game_state:
        :param depth:
        :param ghost_index:
        :return:
        """
        if game_state.isLose() or game_state.isWin():
            return game_state.getScore()

        score = - np.inf
        pacman_index = 0
        actions = game_state.getLegalActions(agentIndex=ghost_index)
        next_agent_index = (ghost_index + 1) % game_state.getNumAgents()

        for action in actions:
            prob = 1.0 / len(actions)
            if not np.isfinite(score):
                score = 0

            if next_agent_index == pacman_index:
                if depth == self.depth - 1:
                    a_ = self.evaluationFunction(
                        game_state.generateSuccessor(agentIndex=ghost_index, action=action))
                    score += prob * a_

                else:
                    score += prob * self.max_agent_decision(
                        game_state=game_state.generateSuccessor(agentIndex=ghost_index, action=action),
                        depth=depth + 1,
                    )
            else:
                score += prob * self.min_agent_score(
                    game_state=game_state.generateSuccessor(agentIndex=ghost_index, action=action),
                    depth=depth,
                    ghost_index=next_agent_index,
                )

        return score


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Useful information you can extract from a GameState (pacman.py)
    pacman_pos = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()
    foods = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    "*** YOUR CODE HERE ***"
    ##########################
    # FOOD SCORE
    ##########################
    foods_positions = matrixAsList(matrix=foods.data, value=True)
    nearest_food = min([manhattanDistance(xy1=pacman_pos, xy2=food) for food in foods_positions])

    ##########################
    # CAPSULE POS
    ##########################
    dist_pacman_capsule = [manhattanDistance(xy1=pacman_pos, xy2=capsule) for capsule in capsules]
    if len(dist_pacman_capsule) > 0:
        nearest_capsule = min(dist_pacman_capsule)
    else:
        nearest_capsule = 0

    ##########################
    # GHOST DISTANCE
    ##########################
    ghosts_positions = [ghost_state.configuration.pos for ghost_state in ghost_states]
    dist_pacman_ghosts = [manhattanDistance(xy1=pacman_pos, xy2=ghost) for ghost in ghosts_positions]
    nearest_ghost_dist = min(dist_pacman_ghosts)

    ##########################
    # TOTAL SCORE
    ##########################

    mu = 0 * np.mean([nearest_ghost_dist, nearest_food, nearest_capsule])
    sigma = np.std([nearest_ghost_dist, nearest_food, nearest_capsule])
    if sigma == 0:
        sigma = 1
    x = (nearest_ghost_dist - mu) / sigma
    y = (nearest_food - mu) / sigma
    z = (nearest_capsule - mu) / sigma

    score = x - y - z + currentGameState.getScore()

    return score


# Abbreviation
better = betterEvaluationFunction
