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

    def check_row(self, board, row, col, row_len, col_len):
        res = 0
        if row == 0 and col == 0:
            # (0,0) with (0,1) or (1,0)
            if board[row][col] > 0 and (
                    board[row][col] == board[row + 1][col] or board[row][col] == board[row][
                col + 1]):
                res += board[row][col]
            return res, True
        if row == 0 and col == col_len - 1:
            # (0,n-1) with (1,n-1) or (0,n-2)
            if (board[row][col_len - 1] == board[row + 1][col_len - 1] or board[row][col] ==
                board[0][col - 1]) and \
                    board[row][col] > 0:
                res += board[row][col_len - 1]
            return res, True
        if row == 0:  # j>0
            # (0,j) with (0,j-1) or (i+1,j) or (0,j+1) // j>0
            if (board[row][col] == board[row][col - 1] or board[row][col] == board[row][
                col + 1] or board[row + 1][col] == board[row][col]) and board[row][col] > 0:
                res += board[row][col]
            return res, True

        if row == row_len - 1 and col == 0:
            # (n-1,0) with (n-2,0) or (n-1,1)
            if board[row][col] > 0 and (
                    board[row][col] == board[row - 1][col] or board[row][col] == board[row][
                col + 1]):
                res += board[row][col]
            return res, True

        if row == row_len - 1 and col == col_len - 1:
            # (n-1,n-1) with (n-2,n-1) or (n-1,n-2)
            if board[row][col] > 0 and (
                    board[row][col] == board[row - 1][col] or board[row][col] == board[row][
                col - 1]):
                res += board[row][col]
            return res, True

        if row == row_len - 1:  # j>0
            # (n-1,j) with (n-1,j-1) or (n-1,j+1)
            if board[row][col] > 0 and (
                    board[row][col] == board[row][col - 1] or board[row][col] == board[row][
                col + 1] or board[row][col] == board[row - 1][col]):
                res += board[row][col]
            return res, True
        return 0, False

    def check_col(self, board, row, col, col_len):
        res = 0
        if col == 0:
            # (i,0) with (i+1,j) or (i-1,j) or (i,j+1)
            if board[row][col] > 0 and (
                    board[row][col] == board[row + 1][col] or board[row][col] == board[row - 1][
                col] or
                    board[row][col] == board[row][col + 1]):
                res += board[row][col]
            return res, True
        if col == col_len - 1:
            # (i,n-1) with (i+1,n-1) or (i-1,n-1) or (i,n-2)
            if board[row][col] > 0 and (
                    board[row][col] == board[row + 1][col] or board[row][col] ==
                    board[row - 1][col] or board[row][col] == board[row][col - 1]):
                res += board[row][col]
            return res, True
        return 0, False

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
        row_len = len(board)
        col_len = len(board[0])
        for k in range(row_len * col_len):
            row = k // col_len
            col = k % col_len
            row_check = self.check_row(board, row, col, row_len, col_len)
            if row_check[1]:
                score += row_check[0]
                continue
            col_check = self.check_col(board, row, col, col_len)
            if col_check[1]:
                score += col_check[0]
                continue
            # general case
            if board[row][col] > 0 and (
                    board[row][col] == board[row + 1][col] or board[row][col] == board[row - 1][
                col] or board[row][col] == board[row][col - 1] or board[row][col] == board[row][
                        col + 1]):
                score += board[row][col]

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
        best_action = None
        best_score = -math.inf

        for ac in game_state.get_legal_actions(0):
            tmp = game_state.generate_successor(0, ac)
            cost = self.minmax(tmp, 2 * self.depth - 1, 1)
            if cost > best_score:
                best_score = cost
                best_action = ac
        return best_action

    def minmax(self, game_state, depth, turn):
        legal_moves = game_state.get_legal_actions(turn)
        legal_moves_length = len(legal_moves)
        if depth == 0 or len(legal_moves) == 0:
            return self.evaluation_function(game_state)
        if turn == 0:
            highest_score = -math.inf
            for i in range(legal_moves_length):
                this_move = game_state.generate_successor(turn, legal_moves[i])
                score = self.minmax(this_move, depth - 1, -1 * turn + 1)
                if highest_score < score:
                    highest_score = score
            return highest_score
        else:
            lowest_score = math.inf
            for i in range(legal_moves_length):
                this_move = game_state.generate_successor(turn, legal_moves[i])
                score = self.minmax(this_move, depth - 1, -1 * turn + 1)
                if score < lowest_score:
                    lowest_score = score
            return lowest_score


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
        alpha = -math.inf
        beta = math.inf
        for ac in game_state.get_legal_actions(0):
            tmp = game_state.generate_successor(0, ac)
            cost = self.alpha_beta_search(tmp, 2 * self.depth - 1, 1, alpha, beta)
            if cost > alpha:
                alpha = cost
                action_max = ac

        return action_max

    def alpha_beta_search(self, game_state, depth, turn, alpha, beta):
        legal_moves = game_state.get_legal_actions(turn)
        legal_moves_length = len(legal_moves)
        if depth == 0 or len(legal_moves) == 0:
            return self.evaluation_function(game_state)
        if turn == 0:
            for i in range(legal_moves_length):
                this_move = game_state.generate_successor(turn, legal_moves[i])
                alpha = max(alpha,
                            self.alpha_beta_search(this_move, depth - 1, -1 * turn + 1, alpha,
                                                   beta))
                if beta <= alpha:
                    break
            return alpha

        else:
            for i in range(legal_moves_length):
                this_move = game_state.generate_successor(turn, legal_moves[i])
                beta = min(beta,
                           self.alpha_beta_search(this_move, depth - 1, -1 * turn + 1, alpha, beta))
                if beta <= alpha:
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
        best_action = None
        best_score = -math.inf
        for ac in game_state.get_legal_actions(0):
            tmp = game_state.generate_successor(0, ac)
            cost = self.expectimax_search(tmp, 2 * self.depth - 1, 1, best_score)
            if cost > best_score:
                best_score = cost
                best_action = ac
        return best_action

    def expectimax_search(self, game_state, depth, turn, best_score):
        legal_moves = game_state.get_legal_actions(turn)
        legal_moves_length = len(legal_moves)
        if depth == 0 or len(legal_moves) == 0:
            return self.evaluation_function(game_state)
        if turn == 0:
            for i in range(legal_moves_length):
                this_move = game_state.generate_successor(turn, legal_moves[i])
                best_score = max(best_score,
                                 self.expectimax_search(this_move, depth - 1, -1 * turn + 1,
                                                        best_score))
            return best_score

        else:
            score = 0
            for i in range(legal_moves_length):
                this_move = game_state.generate_successor(turn, legal_moves[i])
                score += self.expectimax_search(this_move, depth - 1, -1 * turn + 1,
                                                best_score) / len(
                    legal_moves)
            return score


def get_rotated_board(board):
    """
    Return rotated view such that the action is RIGHT.
    """
    rotated_board = board
    rotated_board = rotated_board[:, -1::-1]

    return rotated_board


def smoothness(board):
    """measure difference between tiles and minimize it"""
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

    DESCRIPTION: So we activate the smoothness function to calculate the difference between
    neighbour tiles and minimize it, we also calculate the monotonicity in all directions
    then we sum the score all together
    """
    "*** YOUR CODE HERE ***"
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score
    empty_cell = 0
    if np.count_nonzero(board) != 16:
        empty_cell = np.log(16 - np.count_nonzero(board))
    weight = {"smooth": 0.1, "mono": 0.5, "empty": 2.7, "max_tile": 1}
    smooth = smoothness(board)
    return monotonicity(current_game_state) * weight["mono"] + max_tile * weight[
        "max_tile"] + empty_cell * \
           weight["empty"] + \
           weight["smooth"] * smooth


def monotonicity(current_game_state):
    board = current_game_state.board
    best = -1
    for i in range(1, 4):
        current = 0
        for row in range(4):
            for col in range(3):
                if board[row][col] >= board[row][col + 1]:
                    current += board[row][col]
        for col in range(4):
            for row in range(3):
                if board[row][col] >= board[row + 1][col]:
                    current += board[row][col]

        if current > best:
            best = current
        board = get_rotated_board(board)
    return best


def best_function(current_game_state):
    """ our function iterates over the the successors and gets the highest score according
    to the given weighted score with the snake way"""
    weight = [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]]
    board = current_game_state.board
    board_x = len(current_game_state.board)
    board_y = len(current_game_state.board[0])
    successor_score = 0
    best = -1
    succ_actions = current_game_state.get_legal_actions(0)
    if not succ_actions:
        for i in range(board_x):
            for j in range(board_y):
                if board[i][j] > 0:
                    successor_score += board[i][j] * weight[i][j]
        return successor_score
    for action in succ_actions:
        successor_game_state = current_game_state.generate_successor(action=action)
        for i in range(board_x):
            for j in range(board_y):
                if successor_game_state.board[i][j] > 0:
                    successor_score += successor_game_state.board[i][j] * weight[i][j]
        if best < successor_score:
            best = successor_score
        successor_score = 0

    return best


# Abbreviation
better = better_evaluation_function
