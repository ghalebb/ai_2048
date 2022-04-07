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
        action_max = None
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
        ls_action = game_state.get_legal_actions(turn)
        if depth == 0 or len(ls_action) == 0:
            return self.evaluation_function(game_state)
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
        action_max = None
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
        ls_action = game_state.get_legal_actions(turn)
        if depth == 0 or len(ls_action) == 0:
            return self.evaluation_function(game_state)
        if turn == 0:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                alfa = max(alfa, self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa, beta))
                if beta <= alfa:
                    break
            return alfa

        else:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                beta = min(beta, self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa, beta))
                if beta <= alfa:
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
        action_max = None
        alfa = -math.inf
        for ac in game_state.get_legal_actions(0):
            tmp = game_state.generate_successor(0, ac)
            cost = self.get_action_r(tmp, 2 * self.depth - 1, 1, alfa)
            if cost > alfa:
                alfa = cost
                action_max = ac

        return action_max

    def get_action_r(self, game_state, depth, turn, alfa):
        ls_action = game_state.get_legal_actions(turn)
        if depth == 0 or len(ls_action) == 0:
            return self.evaluation_function(game_state)
        if turn == 0:
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                alfa = max(alfa, self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa))
            return alfa

        else:
            score = 0
            for i in range(len(ls_action)):
                this_move = game_state.generate_successor(turn, ls_action[i])
                score += self.get_action_r(this_move, depth - 1, -1 * turn + 1, alfa) / len(
                    ls_action)
            return score


def get_rotated_board(board):
    """
    Return rotated view such that the action is RIGHT.
    """
    rotated_board = board
    rotated_board = rotated_board[:, -1::-1]

    return rotated_board

def smoothness(board):
    """Smoothness heuristic measures the difference between neighboring tiles and tries to minimize this count"""
    smoothness = 0

    row, col = len(board), len(board[0]) if len(board) > 0 else 0
    for r in board:
        for i in range(col - 1):
            smoothness -= abs(r[i] - r[i + 1])
            pass
    for j in range(row):
        for k in range(col - 1):
            smoothness -= abs(board[k][j] - board[k + 1][j])

    return smoothness

def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score
    empty_cell = 16 - np.count_nonzero(board)
    weight = {"smooth": 0.1, "mono": 1, "empty": 2.7, "max_tile": 1}
    smooth = smoothness(board)
    best = -1
    for i in range(1, 4):
        current = 0
        for row in range(3):
            for col in range(2):
                if board[row][col] >= board[row][col + 1]:
                    current += board[row][col]

        for col in range(3):
            for row in range(2):
                if board[row][col] >= board[row+1][col]:
                    current += board[row][col]

        if current > best:
            best = current
        board = get_rotated_board(board)

    return best + score


def mono(current_game_state):
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score

    totals = [0, 0, 0, 0]
    for x in range(4):
        current = 0
        next = current + 1
        while next < 4:
            while next < 4 and board[x][next] > 0:
                next += 1
            if next >= 4:
                next -= 1
            if board[x][current] > 0:
                current_value = math.log(board[x][current]) / math.log(2)
            else:
                current_value = 0

            if board[x][next] > 0:
                next_value = math.log(board[x][next]) / math.log(2)
            else:
                next_value = 0

            if current_value > next_value:
                totals[0] += next_value - current_value
            elif next_value > current_value:
                totals[1] += current_value - next_value
            current = next
            next += 1
    for y in range(4):
        current = 0
        next = current + 1
        while next < 4:
            while next < 4 and board[next][y] > 0:
                next += 1
            if next >= 4:
                next -= 1
            if board[current][y] > 0:
                current_value = math.log(board[current][y]) / math.log(2)
            else:
                current_value = 0

            if board[next][y] > 0:
                next_value = math.log(board[next][y]) / math.log(2)
            else:
                next_value = 0

            if current_value > next_value:
                totals[2] += next_value - current_value
            elif next_value > current_value:
                totals[3] += current_value - next_value
            current = next
            next += 1

    return max(totals[0], totals[1]) + max(totals[2], totals[3])


def best_function(current_game_state):
    weight = [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]]

    if not current_game_state.get_legal_actions(0):
        return current_game_state.score

    board_x = len(current_game_state.board)
    board_y = len(current_game_state.board[0])

    best = -1
    for action in current_game_state.get_legal_actions(0):
        successor_game_state = current_game_state.generate_successor(action=action)
        successor_sum = 0
        for i in range(board_x):
            for j in range(board_y):
                if successor_game_state.board[i][j]>0:
                    successor_sum += successor_game_state.board[i][j] * weight[i][j]
        if successor_sum > best:
            best = successor_sum

    return best


# Abbreviation
better = better_evaluation_function
best = best_function
