from games import Game, alphabeta_cutoff_search
import numpy as np
import copy
import games


class Tablut(Game):

    def __init__(self, color):

        self.color = color

        self.camps = (
        ((0, 3), (0, 4), (0, 5), (1, 4)), ((3, 0), (4, 0), (5, 0), (4, 1)), ((8, 3), (8, 4), (8, 5), (7, 4)),
        ((3, 8), (4, 8), (5, 8), (4, 7)))

        self.all_camps = self.camps[0] + self.camps[1] + self.camps[2] + self.camps[3]

        self.castle = (4, 4)

        self.winning = (
        (0, 1), (0, 2), (1, 0), (2, 0), (0, 6), (0, 7), (6, 0), (7, 0), (8, 1), (8, 2), (1, 8), (2, 8), (6, 8), (7, 8),
        (8, 6), (8, 7))

        self.directions = ((0, 1), (0, -1), (1, 0), (-1, 0))  # right left top down

        self.central_cross = ((4, 4), (4, 5), (5, 4), (4, 3), (3, 4))

        self.diagonals = ((0, 3), (1, 2), (2, 1), (3, 0), (5, 0), (6, 1), (7, 2), (8, 3), (8, 5), (7, 6), (5, 7), (4, 8), (3, 8), (2, 7), (1, 6), (0, 5))

        self.previous_states = []

    # we define the possible actions:
    def actions(self, state):
        black_pos = tuple(zip(*np.where(state[1] == 'b')))  # all current black positions
        white_pos = tuple(zip(*np.where(state[1] == 'k'))) + tuple(zip(*np.where(state[1] == 'w')))   # all current white positions + king position
        possible_actions = []

        # we check for which player i have to compute the possible moves
        if state[0] == 'B':  # if it's black's turn we check only the black positions
            pos = black_pos
        else:
            pos = white_pos

        # for every initial position (elements)
        for elements in pos:
            invalid_tiles = white_pos + black_pos + (self.castle,)  # tiles in which we can't go in or pass through apart for the camps
            if elements not in self.all_camps:
                invalid_tiles = invalid_tiles + self.all_camps  # add all camps as invalid positions
            else:  # if a checker is initially in a camp we add only the other camps as invalid positions
                for i, camp_i in enumerate(self.camps):
                    if elements not in camp_i:
                        invalid_tiles = invalid_tiles + camp_i

            # for every direction
            for dire in self.directions:
                checkers = (elements[0] + dire[0], elements[1] + dire[1])  # we compute the final position (checkers) incrementing of 1 in that direction
                while checkers not in invalid_tiles and (checkers[0] <= 8 and checkers[0] >= 0) and (checkers[1] <= 8 and checkers[1] >= 0) and state[1][checkers] == 'e':  # while we don't find an invalid position we go on incrementing
                    possible_actions.append(elements + checkers)
                    checkers = (checkers[0] + dire[0], checkers[1] + dire[1])
        return tuple(possible_actions)

    def result(self, state, move):
        actual_state = copy.deepcopy(state)
        paw = actual_state[1][move[0], move[1]]
        actual_state[1][move[0], move[1]], actual_state[1][move[2], move[3]] = actual_state[1][move[2], move[3]], \
                                                                               actual_state[1][move[0], move[1]]

        # check king on win
        if paw == 'k' and (move[2], move[3]) in self.winning:
            return 'WW', actual_state[1]

        # check eat/end
        for dire in self.directions:
            neighbor = (move[2] + dire[0], move[3] + dire[1])
            if neighbor[0] not in range(9) or neighbor[1] not in range(9):
                continue
            # black eat white
            if paw == 'b' and actual_state[1][neighbor] == 'w':
                super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                if super_neighbor[0] not in range(9) or super_neighbor[1] not in range(9):
                    continue
                if actual_state[1][super_neighbor] == 'b' or super_neighbor in self.all_camps or super_neighbor == self.castle:
                    actual_state[1][neighbor] = 'e'

            # black eat king
            elif paw == 'b' and actual_state[1][neighbor] == 'k':
                super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                if super_neighbor[0] not in range(9) or super_neighbor[1] not in range(9):
                    continue
                if actual_state[1][super_neighbor] == 'w':
                    continue
                nears = [(neighbor[0] + dire1[0], neighbor[1] + dire1[1]) for dire1 in self.directions]
                # king in generic position
                if (actual_state[1][super_neighbor] == 'b' or super_neighbor in self.all_camps) and neighbor not in self.central_cross:
                    return 'BW', actual_state[1]
                # king in castle
                if neighbor == self.castle:
                    if actual_state[1][nears[0]] == 'b' and \
                        actual_state[1][nears[1]] == 'b' and \
                        actual_state[1][nears[2]] == 'b' and \
                        actual_state[1][nears[3]] == 'b':
                        return 'BW', actual_state[1]
                # king near castle
                if self.castle in nears:
                    if (actual_state[1][nears[0]] == 'b' or nears[0] == self.castle) and \
                        (actual_state[1][nears[1]] == 'b' or nears[1] == self.castle) and \
                        (actual_state[1][nears[2]] == 'b' or nears[2] == self.castle) and \
                        (actual_state[1][nears[3]] == 'b' or nears[3] == self.castle):
                        return 'BW', actual_state[1]

            # white/king eat black
            elif (paw == 'w' or paw == 'k') and actual_state[1][neighbor] == 'b':
                super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                if super_neighbor[0] not in range(9) or super_neighbor[1] not in range(9):
                    continue
                if actual_state[1][super_neighbor] == 'w' or actual_state[1][super_neighbor] == 'k' or super_neighbor in self.all_camps or super_neighbor == self.castle:
                    actual_state[1][neighbor] = 'e'

        # change turn
        if actual_state[0] == 'B':
            return 'W', actual_state[1]
        if actual_state[0] == 'W':
            return 'B', actual_state[1]

    # return true if there is a winning condition
    def terminal_test(self, state):
        if state[0] == 'BW' or state[0] == 'WW':
            return True
        return False

    # return the player that has to move
    def to_move(self, state):
        #print('muove il: ',state[0])
        return state[0]


    # taken a winning condition and a player return a score
    def utility(self, state, player):
        if state[0] == 'BW' and player == 'B':
            return 1000000
        elif state[0] == 'WW' and player == 'B':
            return -1000000
        elif state[0] == 'BW' and player == 'W':
            return -1000000
        elif state[0] == 'WW' and player == 'W':
            return 1000000


# Return the number of free line of the king
    def free_king_line(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))[0]
        check = 0
        for dire in self.directions:
            checkers = (k_pos[0] + dire[0], k_pos[1] + dire[1])
            if checkers[0] not in range(9) or checkers[1] not in range(9):
                continue
            while state[1][checkers] == 'e' and checkers not in self.all_camps and checkers != self.castle:
                if checkers in self.winning:
                    check += 1
                    break
                checkers = (checkers[0] + dire[0], checkers[1] + dire[1])
        if check>1:
            return 100000
        return check

    # Return the number of free line in the board
    def free_line(self, state):
        count = 0
        for i in (2, 6):
            for j in range(9):
                if state[1][i, j] == 'b' or state[1][i, j] == 'w':
                    break
            if j == 8 and (state[1][i, j] != 'b' and state[1][i, j] != 'w'):
                count += 1
        for i in (2, 6):
            for j in range(9):
                if (state[1][j, i] == 'b' or state[1][j, i] == 'w'):
                    break
            if j == 8 and state[1][j, i] != 'b' and state[1][j, i] != 'w':
                count += 1
        return count

    # Return the number of black pawns near king
    def near_king(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        n_black = 0
        for dire in self.directions:
            if 0 <= k_pos[0] + dire[0] <= 8 and 0 <= k_pos[1] + dire[1] <= 8:
                if state[1][k_pos[0]+dire[0], k_pos[1]+dire[1]] == 'b':
                    n_black += 1
        return n_black

    # Return the number of black pawns in the 4 diagonals we have defined
    def diag_b(self, state):
        count = 0
        for d in self.diagonals:
            if state[1][d] == 'b':
                count += 1
        return count

    def diag_w(self, state):
        count = 0
        for d in self.diagonals:
            if state[1][d] == 'w':
                count += 1

    # Return number of black pawns
    def black_pawns(self, state):
        black = np.where(state[1] == 'b')
        black = tuple(zip(black[0], black[1]))
        n_black = len(black)
        return n_black


    # Return number of white pawns
    def white_pawns(self, state):
        white = np.where(state[1] == 'w')
        white = tuple(zip(white[0], white[1]))
        n_white = len(white)
        return n_white


    # Return true if a black pawns threats the king (not used)
    def king_threat(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        counter = 0
        edir = ()
        if k_pos in self.central_cross:
            for dire in self.directions:
                if (k_pos[0] + dire[0], k_pos[1] + dire[1]) == self.castle or state[1][k_pos[0] + dire[0], k_pos[1] + dire[1]] == 'b':
                    counter += 1
                else:
                    edir = dire
            if counter == 3:
                new_pos = (k_pos[0] + edir[0], k_pos[1] + edir[1])
                for dire in self.directions:
                    if dire[0] == -(edir[0]) and dire[1] == -(edir[1]):
                        continue
                    else:
                        new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                        while 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8 and state[1][new_pos] == 'e':
                            new_pos = (k_pos[0] + edir[0], k_pos[1] + edir[1])
                        if state[1][new_pos] == 'b':
                            return True
            return False
        for dire in self.directions:
            new_pos = (k_pos[0] + dire[0], k_pos[1] + dire[1])
            if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                if state[1][new_pos] == 'b' or new_pos in self.all_camps:
                    new_pos = (new_pos[0] - 2*dire[0], new_pos[1] - 2*dire[1])
                    if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                        for dir1 in self.directions:
                            if dir1 != dire:
                                new_pos1 = (new_pos[0], new_pos[1])
                                while new_pos1 != self.castle and 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and state[1][new_pos1] == 'e':
                                    if new_pos1 in self.all_camps:
                                        if (new_pos1[0] + dir1[0], new_pos1[1] + dir1[1]) not in self.all_camps:
                                            break
                                    new_pos1 = (new_pos1[0] + dir1[0], new_pos1[1] + dir1[1])
                                if 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and state[1][new_pos1] == 'b' and new_pos1 != new_pos:
                                    return True
        return False

    # Return the number of white checkers under threat (not used)
    def white_threat(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        count = 0
        for white in white_pos:
            for dire in self.directions:
                new_pos = (white[0] + dire[0], white[1] + dire[1])
                if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                    if (state[1][new_pos] == 'b') or (new_pos in self.all_camps) or (new_pos == self.castle):
                        new_pos = (new_pos[0] - 2 * dire[0], new_pos[1] - 2 * dire[1])
                        if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                            for dir1 in self.directions:
                                if dir1 != dire:
                                    new_pos1 = (new_pos[0], new_pos[1])
                                    while new_pos1 != self.castle and 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and state[1][new_pos1] == 'e':
                                        if new_pos1 in self.all_camps:
                                            if state[1][new_pos1] == 'b':
                                                break
                                            if (new_pos1[0] + dir1[0], new_pos1[1] + dir1[1]) not in self.all_camps:
                                                break
                                        new_pos1 = (new_pos1[0] + dir1[0], new_pos1[1] + dir1[1])
                                    if 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and state[1][new_pos1] == 'b' and new_pos1 != new_pos:
                                        count += 1
        return count

    # Return the number of black under threat (not used)
    def black_threat(self, state):
        black_pos = np.where(state[1] == 'b')
        black_pos = tuple(zip(black_pos[0], black_pos[1]))
        count = 0
        for black in black_pos:
            if black not in self.all_camps:
                for dire in self.directions:
                    new_pos = (black[0] + dire[0], black[1] + dire[1])
                    if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                        if state[1][new_pos] == 'w' or new_pos in self.all_camps or new_pos == self.castle:
                            new_pos = (new_pos[0] - 2 * dire[0], new_pos[1] - 2 * dire[1])
                            if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                                for dir1 in self.directions:
                                    if dir1 != dire:
                                        new_pos1 = (new_pos[0], new_pos[1])
                                        while new_pos1 != self.castle and new_pos1 not in self.all_camps and 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and state[1][new_pos1] == 'e':
                                            new_pos1 = (new_pos1[0] + dir1[0], new_pos1[1] + dir1[1])
                                        if 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and (state[1][new_pos1] == 'w' or state[1][new_pos1] == 'k') and new_pos1 != new_pos:
                                            count += 1
                                            break
            elif black in ((1, 4), (4, 1), (7, 4), (4, 7)):
                for dire in self.directions:
                    new_pos = (black[0] + dire[0], black[1] + dire[1])
                    if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                        if state[1][new_pos] == 'w':
                            new_pos = (new_pos[0] - 2 * dire[0], new_pos[1] - 2 * dire[1])
                            if 0 <= new_pos[0] <= 8 and 0 <= new_pos[1] <= 8:
                                for dir1 in self.directions:
                                    if dir1 != dire:
                                        new_pos1 = (new_pos[0], new_pos[1])
                                        while new_pos1 not in self.all_camps and 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and state[1][new_pos1] == 'e':
                                            new_pos1 = (new_pos1[0] + dir1[0], new_pos1[1] + dir1[1])
                                        if 0 <= new_pos1[0] <= 8 and 0 <= new_pos1[1] <= 8 and (state[1][new_pos1] == 'w' or state[1][new_pos1] == 'k') and new_pos1 != new_pos:
                                            count += 1
                                            break
        return count

    #return true if the king is outside the castle
    def king_outside_castle(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        if k_pos != self.castle:
            return True
        return False

    # return the number of black pawns in king's quadrant (not used)
    def black_pawns_in_king_quad(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        n_black = 0
        if k_pos[0] > 4:
            if k_pos[1] > 4:
                matrix = np.array(state[1][5:, 5:])
                black_pos = np.where(matrix == 'b')
                black_pos = tuple(zip(black_pos[0], black_pos[1]))
                n_black = len(black_pos)
            elif k_pos[1] < 4:
                matrix = np.array(state[1][5:, :4])
                black_pos = np.where(matrix == 'b')
                black_pos = tuple(zip(black_pos[0], black_pos[1]))
                n_black = len(black_pos)
        elif k_pos[0] < 4:
            if k_pos[1] > 4:
                matrix = np.array(state[1][:4, 5:])
                black_pos = np.where(matrix == 'b')
                black_pos = tuple(zip(black_pos[0], black_pos[1]))
                n_black = len(black_pos)
            elif k_pos[1] < 4:
                matrix = np.array(state[1][:4, :4])
                black_pos = np.where(matrix == 'b')
                black_pos = tuple(zip(black_pos[0], black_pos[1]))
                n_black = len(black_pos)
        #print('nuero di neri nel quadrante del re: ',n_black)
        return n_black

    # return the number of black pawns in the 2 quadrants near the king's quadrant (not used)
    def black_pawns_in_king_near_quad(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        n_black = 0
        if k_pos[0] > 4:
            if k_pos[1] > 4:
                matrix1 = np.array(state[1][5:, :4])
                matrix2 = np.array(state[1][:4, 5:])
                black_pos1 = np.where(matrix1 == 'b')
                black_pos2 = np.where(matrix2 == 'b')
                black_pos1 = tuple(zip(black_pos1[0], black_pos1[1]))
                black_pos2 = tuple(zip(black_pos2[0], black_pos2[1]))
                n_black = len(black_pos1) + len(black_pos2)
            elif k_pos[1] < 4:
                matrix1 = np.array(state[1][:4, :4])
                matrix2 = np.array(state[1][5:, 5:])
                black_pos1 = np.where(matrix1 == 'b')
                black_pos2 = np.where(matrix2 == 'b')
                black_pos1 = tuple(zip(black_pos1[0], black_pos1[1]))
                black_pos2 = tuple(zip(black_pos2[0], black_pos2[1]))
                n_black = len(black_pos1) + len(black_pos2)
        elif k_pos[0] < 4:
            if k_pos[1] < 4:
                matrix1 = np.array(state[1][5:, :4])
                matrix2 = np.array(state[1][:4, 5:])
                black_pos1 = np.where(matrix1 == 'b')
                black_pos2 = np.where(matrix2 == 'b')
                black_pos1 = tuple(zip(black_pos1[0], black_pos1[1]))
                black_pos2 = tuple(zip(black_pos2[0], black_pos2[1]))
                n_black = len(black_pos1) + len(black_pos2)
            elif k_pos[1] > 4:
                matrix1 = np.array(state[1][:4, :4])
                matrix2 = np.array(state[1][5:, 5:])
                black_pos1 = np.where(matrix1 == 'b')
                black_pos2 = np.where(matrix2 == 'b')
                black_pos1 = tuple(zip(black_pos1[0], black_pos1[1]))
                black_pos2 = tuple(zip(black_pos2[0], black_pos2[1]))
                n_black = len(black_pos1) + len(black_pos2)
        return n_black

    # return the number of black pawns whe the king is in the central cross (not used)
    def black_pawns_when_king_in_cross(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        n_black = 0
        if k_pos[0] != k_pos[1] and k_pos[0] == 4:
            matrix = np.array(state[1][:, :4])
            black_pos = np.where(matrix == 'b')
            black_pos = tuple(zip(black_pos[0], black_pos[1]))
            n_black = len(black_pos)
        elif k_pos[0] != k_pos[1] and k_pos[0] == 4:
            matrix = np.array(state[1][:4, :])
            black_pos = np.where(matrix == 'b')
            black_pos = tuple(zip(black_pos[0], black_pos[1]))
            n_black = len(black_pos)
        return n_black

    # return the number of whites in the angles (not used)
    def n_white_in_angle(self, state):
        n_white = 0
        if state[1][0, 0] == 'w':
            n_white += 1
        if state[1][8, 0] == 'w':
            n_white += 1
        if state[1][0, 8] == 'w':
            n_white += 1
        if state[1][8, 8] == 'w':
            n_white += 1
        return n_white

    # return the number of white pawns in the victory positions
    def n_white_in_victory(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        n_white = 0
        for white in white_pos:
            if white in self.winning:
                n_white += 1
        return n_white

    # return the number of white pawns in the victory positions with a white near them (not used)
    def n_white_in_victory_near_white(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        n_white = 0
        for white in white_pos:
            if white in self.winning:
                if ((white[0] + 2 <= 8 and state[1][white[0] + 2, white[1]] == 'w') or (white[1] + 2 <= 8 and state[1][white[0], white[1] + 2] == 'w') or (white[0] - 2 >= 0 and state[1][white[0] - 2, white[1]] == 'w') or (white[1] - 2 >= 0 and state[1][white[0], white[1] - 2] == 'w')):
                    n_white += 1
        return n_white

    # return the list of possible king's actions
    def king_actions(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))[0]
        possible_actions = []

        for dire in self.directions:
            checkers = (k_pos[0] + dire[0], k_pos[1] + dire[
                1])  # we compute the final position (checkers) incrementing of 1 in that direction
            while checkers not in self.all_camps and 8 >= checkers[0] >= 0 and 8 >= checkers[1] >= 0 and state[1][
                checkers] == 'e':  # while we don't find an invalid position we go on incrementing
                possible_actions.append(k_pos + checkers)
                checkers = (checkers[0] + dire[0], checkers[1] + dire[1])
        return tuple(possible_actions)

    # return the moves that the king needs to reach the victory (up to 5) (not used)
    def move_to_winning(self, state):
        if self.free_king_line(state) > 0:
            return 1

        next_state1 = []
        for action in self.king_actions(state):
            next_state1.append(self.result(state,action))
        if len(next_state1) == 0:
            return 5
        for state1 in next_state1:
            if self.free_king_line(state1) > 0:
                return 2

        next_state2 = []
        for state1 in next_state2:
            for action2 in self.king_actions(state1):
                next_state2.append(self.result(state1, action2))

        for state2 in next_state2:
            if self.free_king_line(state2) > 0:
                return 3

        return 5

    # evakuation function that return us a score for a given state
    def white_evaluation_function(self, state):
        if self.terminal_test(state):
            return self.utility(state, self.color)
        if self.color == 'W':
            weights = [50, 5, -2, -2, -110, 50, -0, -1, 1.5, 30, -0, -0, -0, 0, 2, 0, -100]
        elif self.color == 'B':
            weights = [-25, -5, 20, 2, 90, -30, 0, 1, -1.5, -5, 0, 0, 0, -0, -2, -0, 0]

        # we decide not to put all the functions because else it would be too heavy
        w1 = self.free_king_line(state) * weights[0]
        w2 = 0#self.free_line(state) * weights[1]
        w3 = self.near_king(state) * weights[2]
        w4 = self.diag_b(state) * weights[3]
        w5 = self.black_pawns(state) * weights[4]
        w6 = self.white_pawns(state) * weights[5]
        w7 = 0#self.king_threat(state) * weights[6]
        w8 = 0#self.white_threat(state) * weights[7]
        w9 = 0#self.black_threat(state) * weights[8]
        w10 = self.king_outside_castle(state) * weights[9]
        w11 = 0#self.black_pawns_in_king_quad(state) * weights[10]
        w12 = 0#self.black_pawns_in_king_near_quad(state) * weights[11]
        w13 = 0#self.black_pawns_when_king_in_cross(state) * weights[12]
        w14 = 0#self.n_white_in_angle(state) * weights[13]
        w15 = self.n_white_in_victory(state) * weights[14]
        w16 = 0#self.n_white_in_victory_near_white(state) * weights[15]
        w17 = 0#self.move_to_winning(state) * weights[16]
        weights_white = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17]
        return sum(weights_white)
