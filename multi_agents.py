import math

import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        "*** YOUR CODE HERE ***"
        row_len = len(board)
        col_len = len(board[0])

        for i in range(row_len):
            for j in range(col_len):
                if i == 0 and j == 0:
                    # (0,0) with (0,1) or (1,0)
                    if (board[i][j] == board[i + 1][j] or board[i][j] == board[i][
                        j + 1]) and \
                            board[i][j] > 0:
                        score += board[i][j]
                    continue
                if i == 0 and j == col_len - 1:
                    # (0,n-1) with (1,n-1) or (0,n-2)
                    if (board[i][col_len - 1] == board[i + 1][col_len - 1] or board[i][j] ==
                        board[0][j - 1]) and \
                            board[i][j] > 0:
                        score += board[i][col_len - 1]
                    continue
                if i == 0:  # j>0
                    # (0,j) with (0,j-1) or (i+1,j) or (0,j+1) // j>0
                    if (board[i][j] == board[i][j - 1] or board[i][j] == board[i][
                        j + 1] or board[i + 1][j] == board[i][j]) and board[i][j] > 0:
                        score += board[i][j]
                    continue
                if i == row_len - 1 and j == 0:
                    # (n-1,0) with (n-2,0) or (n-1,1)
                    if board[i][j] > 0 and (
                            board[i][j] == board[i - 1][j] or board[i][j] == board[i][
                        j + 1]):
                        score += board[i][j]
                    continue
                if i == row_len - 1 and j == col_len - 1:
                    # (n-1,n-1) with (n-2,n-1) or (n-1,n-2)
                    if board[i][j] > 0 and (
                            board[i][j] == board[i - 1][j] or board[i][j] == board[i][
                        j - 1]):
                        score += board[i][j]
                    continue
                if i == row_len - 1:  # j>0
                    # (n-1,j) with (n-1,j-1) or (n-1,j+1)
                    if board[i][j] > 0 and (
                            board[i][j] == board[i][j - 1] or board[i][j] == board[i][
                        j + 1] or board[i][j] == board[i - 1][j]):
                        score += board[i][j]
                    continue
                if j == 0:
                    # (i,0) with (i+1,j) or (i-1,j) or (i,j+1)
                    if board[i][j] > 0 and (
                            board[i][j] == board[i + 1][j] or board[i][j] == board[i - 1][j] or
                            board[i][j] == board[i][j + 1]):
                        score += board[i][j]
                    continue
                if j == col_len - 1:
                    # (i,n-1) with (i+1,n-1) or (i-1,n-1) or (i,n-2)
                    if board[i][j] > 0 and (board[i][j] == board[i + 1][j] or board[i][j] ==
                                            board[i - 1][j] or board[i][j] == board[i][j - 1]):
                        score += board[i][j]
                    continue

                # general case
                if board[i][j] > 0 and (
                        board[i][j] == board[i + 1][j] or board[i][j] == board[i - 1][
                    j] or board[i][j] == board[i][j - 1] or board[i][j] == board[i][j + 1]):
                    score += board[i][j]

        return score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        max_score = 0
        action_max = Action.STOP
        alfa = -math.inf
        beta = math.inf
        for ac in game_state.get_legal_actions(0):
            tmp = game_state.generate_successor(0, ac)
            cost = self.get_action_r(tmp, 2 * self.depth - 1, 1, alfa, beta)
            if cost > alfa:
                alfa = cost
                action_max = ac

        return action_max



    # def get_action_r(self, game_state, depth, turn):
    #     if depth == 0:
    #         return self.evaluation_function(game_state)
    #     ls_action = game_state.get_legal_actions(turn)
    #     ret_score = -1
    #     for i in range(len(ls_action)):
    #         this_move = game_state.generate_successor(turn, ls_action[i])
    #         score = self.get_action_r(this_move, depth - 1, -1 * turn + 1)
    #         if ret_score == -1:
    #             ret_score = score
    #         elif turn == 1 and ret_score < score:
    #             ret_score = score
    #         elif turn == 0 and ret_score > score:
    #             ret_score = score
    #     return ret_score

    def get_action_r(self, game_state, depth, turn, alfa, beta):
        if depth == 0:
            return self.evaluation_function(game_state)
        ls_action = game_state.get_legal_actions(turn)
        if turn == 0:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                score = self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa, beta)
                if alfa < score:
                    alfa = score
            return alfa
        else:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                score = self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa, beta)
                if score < beta:
                    beta = score
            return beta



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        action_max = Action.STOP
        alfa = -math.inf
        beta = math.inf
        for ac in game_state.get_legal_actions(0):
            tmp = game_state.generate_successor(0, ac)
            cost = self.get_action_r(tmp, 2 * self.depth - 1, 1, alfa, beta)
            if cost > alfa:
                alfa = cost
                action_max = ac

        return action_max

    def get_action_r(self, game_state, depth, turn, alfa, beta):
        if depth == 0:
            return self.evaluation_function(game_state)
        ls_action = game_state.get_legal_actions(turn)
        if turn == 0:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                score = self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa, beta)
                if alfa < score:
                    alfa = score
                if beta <= alfa:
                    break
            return alfa

        else:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                score = self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa, beta)
                if score < beta:
                    beta = score
                if beta >= alfa:
                    break
            return beta



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
