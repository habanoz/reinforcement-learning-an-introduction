import random

all_states = {}


def get_winner(board_tuple):
    if board_tuple not in all_states:
        winner = is_end(board_tuple)
        all_states[board_tuple] = winner

    return all_states[board_tuple]


class Player:
    def __init__(self, symbol):
        self.symbol = symbol

        self.epsilon = 0.01
        self.step_size = 0.1

        self.valueEstimates = {}
        self.observation_tuples = []

    def act(self, board_tuple):
        possible_actions_values = [(self.get_estimate_value(board_tuple, action), action) for action in
                                   range(len(board_tuple)) if board_tuple[action] == 0]

        if random.random() < self.epsilon:
            selected_action = random.choice(possible_actions_values)[1]
        else:
            random.shuffle(possible_actions_values)
            possible_actions_values.sort(key=lambda x: x[0])
            selected_action = possible_actions_values[-1][1]  # the last one having the greatest value

        return selected_action

    def add_observation(self, board):
        board_tuple = tuple(board)
        self.observation_tuples.append(board_tuple)

        if board_tuple not in self.valueEstimates:
            self.initialize_value_estimate(board_tuple)

    def reset(self):
        self.observation_tuples = []

    def get_estimate_value(self, board_tuple, action):
        new_board = list(board_tuple)
        new_board[action] = self.symbol
        new_board_tuple = tuple(new_board)

        if new_board_tuple not in self.valueEstimates:
            self.initialize_value_estimate(new_board_tuple)

        return self.valueEstimates[new_board_tuple]

    def initialize_value_estimate(self, board_tuple):
        # initialize
        winner = get_winner(board_tuple)
        if winner:
            if winner == self.symbol:  # win
                self.valueEstimates[board_tuple] = 1
            elif winner == 2:  # draw
                self.valueEstimates[board_tuple] = 0.5
            else:  # lose
                self.valueEstimates[board_tuple] = 0
        else:  # in-complete
            self.valueEstimates[board_tuple] = 0.5

    def backup(self):
        for t in reversed(range(len(self.observation_tuples) - 1)):
            state = self.observation_tuples[t]
            next_state = self.observation_tuples[t + 1]

            td_error = self.valueEstimates[next_state] - self.valueEstimates[state]
            self.valueEstimates[state] += self.step_size * td_error


class HumanPlayer(Player):
    def __init__(self, symbol):
        Player.__init__(self, symbol)

    def act(self, board_tuple):
        print('Board {}'.format(board_tuple))
        action = input('Chose your input')
        return action


class Judger:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    @staticmethod
    def initial_board():
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def peer_turn(self):
        while True:
            yield self.player1
            yield self.player2

    def train(self, epochs, epsilon=0.01, print_every_n=1000):
        player1_win = 0
        player2_win = 0

        self.player1.epsilon = epsilon
        self.player2.epsilon = epsilon

        for i in range(1, epochs + 1):
            winner = self.play()

            self.player1.backup()
            self.player2.backup()

            if winner == 1:
                player1_win += 1
            if winner == -1:
                player2_win += 1
            if i % print_every_n == 0:
                print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f, last winner %d' % (
                    i, player1_win / i, player2_win / i, winner))

    def compete(self, epochs):
        print("Competing for %d epochs" % (epochs))
        self.train(epochs, 0, 1)

    def play(self):
        winner = 0
        board = Judger.initial_board()
        peer = self.peer_turn()

        self.player1.reset()
        self.player2.reset()

        self.player1.add_observation(board)
        self.player2.add_observation(board)

        while not winner:
            player = next(peer)
            action = player.act(tuple(board))
            board[int(action)] = player.symbol

            self.player1.add_observation(board)
            self.player2.add_observation(board)

            winner = is_end(board)

        return winner


def is_end(board):
    is_end_p = is_end_part(board[0:3])
    if is_end_p: return is_end_p

    is_end_p = is_end_part(board[3:6])
    if is_end_p: return is_end_p

    is_end_p = is_end_part(board[6:9])
    if is_end_p: return is_end_p

    # first col
    is_end_p = is_end_part(board[0:9:3])
    if is_end_p: return is_end_p

    #second col
    is_end_p = is_end_part(board[1:9:3])
    if is_end_p: return is_end_p

    # third col
    is_end_p = is_end_part(board[2:9:3])
    if is_end_p: return is_end_p

    # 0 4 8
    is_end_p = is_end_part(board[0:9:4])
    if is_end_p: return is_end_p

    # 2 4 6
    is_end_p = is_end_part(board[2:7:2])
    if is_end_p: return is_end_p

    if sum(map(abs, board)) == len(board):
        return 2  # draw

    return 0  # not ended


def is_end_part(arr):
    summation = sum(arr)
    if summation == 3:
        return 1

    if summation == -3:
        return -1

    return 0


if __name__ == '__main__':
    player1 = Player(1)
    player2 = Player(-1)
    judger = Judger(player1, player2)
    judger.train(int(5e5))
    judger.compete(int(1e3))

    judger = Judger(player2, player1)
    judger.compete(int(1e3))

    #judger = Judger(player1, HumanPlayer(-1))
    #judger.compete(3)

    judger = Judger(HumanPlayer(-1), player1)
    judger.compete(3)
