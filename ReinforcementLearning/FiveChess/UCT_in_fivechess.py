# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random
from gamegobang import fivechessenv


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = empty, 1 = player 1, 2 = player 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z] != 0:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return -1.0
        if self.GetMoves() == []: return 0.0  # draw
        return -2.0
        assert False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            x = ".XO"[self.board[i]]
            s += "{0:2}".format(x)
            if i % 3 == 2: s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = max(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
            self.untriedMoves) + "]"

    def TreeToString(self, indent, flag=False):
        if flag == True:
            s = self.IndentString(indent) + str(self) + " " + str(
                self.wins / self.visits + sqrt(2 * log(self.visits) / self.parentNode.visits))
        else:
            s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1, flag=True)
            # + str(c.wins / c.visits + 0.8 * sqrt(2 * log(self.visits) / c.visits)) + "  "
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "|   "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose=False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()
        action = [0, 0, 0]
        reward = 0
        over = False

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            # print("playerJustMoved: node: {}, state: {}".format(node.playerJustMoved, state.playerJustMoved))
            state.exchange_player()
            action[0], action[1], action[2] = node.move // state.SIZE, node.move % state.SIZE, state.playerJustMoved
            _, reward, win, _ = state.step(action)
            if win:
                over = True
                # state.reset()

        # Expand
        if node.untriedMoves != [] and not over:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.exchange_player()
            action[0], action[1], action[2] = m // state.SIZE, m % state.SIZE, state.playerJustMoved
            _, reward, win, _ = state.step(action)
            if win:
                over = True
                # state.reset()
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != [] and not over:  # while state is non-terminal
            m = random.choice(state.GetMoves())
            state.exchange_player()
            action[0], action[1], action[2] = m // state.SIZE, m % state.SIZE, state.playerJustMoved
            _, reward, win, _ = state.step(action)
            if win:
                over = True
                # state.reset()

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            # print("Player: {}, Reward: {}".format(node.playerJustMoved, node.playerJustMoved * action[2] * reward))
            # state is terminal. Update node with result from POV of node.playerJustMoved
            node.Update(node.playerJustMoved * action[2] * reward)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if verbose:
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())

    return max(rootnode.childNodes, key=lambda c: c.visits).move  # return the move that was most visited


def UCTPlayGame():
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = fivechessenv.FiveChessEnv()
    height = width = state.SIZE

    while state.GetMoves() != []:

        state.render()
        # Make random action
        action = [0, 0, 0]

        if state.playerJustMoved == 1:
            # Select move for other player
            m = UCT(rootstate=state, itermax=10000, verbose=True)
            action[0], action[1] = m // height, m % width
        else:
            # Human play
            location = input("Player {} input:".format(0 - state.playerJustMoved))
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            action[0], action[1] = location[0], location[1]
            # m = UCT(rootstate=state, itermax=10000, verbose=True)
            # action[0], action[1] = m // height, m % width

        state.exchange_player()
        action[2] = state.playerJustMoved

        print("Best Move for player {}: ({}, {})".format(action[2], action[0], action[1]))
        # Judgement result
        _, _, win, _ = state.step(action)
        if win:
            state.render()
            print("Game done")
            break


if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    UCTPlayGame()
