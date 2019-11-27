from games import Game
import numpy as np
import copy


# stato= numpy array riempito con caratteri che rappresentano vuoto, bianco, nero, re ('e','w','b','k')
class Tablut(Game):

    # definisco la scacchiera:
    # -dimensioni
    # -campi
    # -castello
    # -punti di vittoria
    # stato
    def __init__(self):
        self.initial_state = ('W', np.array([['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                             ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                             ['b', 'b', 'w', 'w', 'k', 'w', 'w', 'b', 'b'],
                                             ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                             ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e']]))

        self.keyboard = np.array([['e', 'w', 'w', 'c', 'c', 'c', 'w', 'w', 'e'],
                                  ['w', 'e', 'e', 'e', 'c', 'e', 'e', 'e', 'e'],
                                  ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                                  ['c', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'c'],
                                  ['c', 'c', 'e', 'e', 'k', 'e', 'e', 'c', 'c'],
                                  ['c', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'c'],
                                  ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                                  ['w', 'e', 'e', 'e', 'c', 'e', 'e', 'e', 'e'],
                                  ['e', 'w', 'w', 'c', 'c', 'c', 'w', 'w', 'e']])

        self.camps = (
        ((0, 3), (0, 4), (0, 5), (1, 4)), ((3, 0), (4, 0), (5, 0), (4, 1)), ((8, 3), (8, 4), (8, 5), (7, 4)),
        ((3, 8), (4, 8), (5, 8), (4, 7)))

        self.all_camps = self.camps[0] + self.camps[1] + self.camps[2] + self.camps[3]

        self.castle = (4, 4)

        self.winning = (
        (0, 1), (0, 2), (1, 0), (2, 0), (0, 6), (0, 7), (6, 0), (7, 0), (8, 1), (8, 2), (1, 8), (2, 8), (6, 8), (7, 8),
        (8, 6), (8, 7))

        self.dimension = (9, 9)

        self.directions = ((0, 1), (0, -1), (1, 0), (-1, 0))  # su giù destra sinistra

        self.central_cross = ((4, 4), (4, 5), (5, 4), (4, 3), (3, 4))

        self.n_checkers = (8,16)

        self.white = ((2, 4), (3, 4), (4, 2), (4, 3), (4, 5), (4, 6), (5, 4), (6, 4), (4, 4))

        self.black = ((0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (3, 8), (4, 0), (4, 1), (4, 7), (4, 8), (5, 0), (5, 8), (7, 4), (8, 3), (8, 4), (8, 5))

        self.previous_states = []  #maybe we will initialize with initial state (depending when we call draw)

    # definisco le azioni possibili:
    # -struttura ((x,y),('sopra',1),'num2,num3,num4) or dictionary(key=(pos_X,pos_Y),value=[(pos_X_finale,pos_Y_finale),(..)...lista delle possibili posizioni finali..]
    # le azioni possibili le passiamo come un generatore
    def actions(self, state):

        black_pos = tuple(zip(*np.where(state[1] == 'b')))  # all current black positions
        white_pos = tuple(zip(*np.where(state[1] == 'w'))) + tuple(
            zip(*np.where(state[1] == 'k')))  # all current white positions + king position
        possible_actions = []
        invalid_tiles = white_pos + black_pos + self.castle  # tiles in which we can't go in or pass through apart for the camps

        # we check for which player i have to compute the possible moves
        if state[0] == 'B':  # if it's black's turn we check only the black positions
            pos = black_pos
        else:
            pos = white_pos

        # for every initial position (elements)
        for elements in pos:
            if elements not in self.all_camps:
                invalid_tiles = invalid_tiles + self.all_camps  # add all camps as invalid positions
            else:  # if a checker is initially in a camp we add only the other camps as invalid positions
                for i, camp_i in enumerate(self.camps):
                    if elements in camp_i:
                        for j in range(4):
                            if j != i:
                                invalid_tiles = invalid_tiles + self.camps[j]
                        break  # once we find a checkers in a camp we don't check the other camps for it

            # for every direction
            for dire in self.directions:
                checkers = (elements[0] + dire[0], elements[1] + dire[
                    1])  # we compute the final position (checkers) incrementing of 1 in that direction
                while checkers not in invalid_tiles:  # while we don't find an invalid position we go on incrementing
                    possible_actions.append((elements, checkers))
                    checkers = (checkers[0] + dire[0], checkers[1] + dire[1])
        return tuple(possible_actions)

    # definiamo lo stato successivo
    ####### bisogna modificare il fatto che non mangia se schiaccio un bianco contro il castello e dentro c'è il re o
    ####### se schiaccio un nero contro il campo e dentro il campo c'è un nero
    def result(self, state, move):
        state1 = copy.deepcopy(state)
        white_pos = list(self.white)
        black_pos = list(self.black)
        if state1[0] == 'B':  # swap turn
            state1[0] = 'W'
            other, my = 'w', 'b'
            black_pos.append(move[1])
            black_pos.remove(move[0])

        else:
            state1[0] = 'B'
            other, my = 'b', 'w'
            white_pos.append(move[1])
            white_pos.remove(move[0])

        state1[1][move[0]], state1[1][move[1]] = state1[1][move[1]], state1[1][
            move[0]]  # swap the current position with an empty tile

        for dire in self.directions:
            neighbor = (move[1][0] + dire[0], move[1][1] + dire[1])
            if state1[1][neighbor] == other:
                super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                if state1[1][super_neighbor] == my or state1[1][super_neighbor] in self.all_camps or state1[1][super_neighbor] == self.castle:
                    state1[1][neighbor] = 'e'
                    if state1[0] == 'B':
                        black_pos.remove(neighbor)
                    else:
                        white_pos.remove(neighbor)
        self.black = tuple(black_pos)
        self.white = tuple(white_pos)
        current_checkers = (len(self.white), len(self.black))
        if self.n_checkers == current_checkers:
            self.previous_states.append(state1[1])
        else:
            self.previous_states = [state1[1]]
            self.n_checkers = current_checkers
        return state1

    # ritorna True quando la scacchiera è in una condizione di vittoria per uno dei due giocaotri
    def terminal_test(self, state, action):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[
            1]))  # mi restituisce una tupla con dentro una tupla che sono la x e la y del king perchè se non metto tuple mi restituirebbe un zip object
        k_pos = k_pos[0]
        near = []
        for dire in self.directions:
            supp = (k_pos[0] + dire[0], k_pos[1] + dire[1])
            near.append(supp)
        if k_pos in self.winning:
            return True
        if k_pos in self.central_cross:
            if ((state[1][k_pos[0] + 1, k_pos[1]] == 'b' or (k_pos + 1, k_pos[1]) == self.castle) \
                    and (state[1][k_pos[0] - 1, k_pos[1]] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) \
                    and (state[1][k_pos[0], k_pos[1] - 1] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) \
                    and (state[1][k_pos[0], k_pos[1] + 1] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) and state[0]=='W'):
                if action[1] in near:
                    return True
        else:
            if (state[1][k_pos[0] + 1, k_pos[1]] == 'b' or (k_pos[0] + 1, k_pos[1]) in self.all_camps) and (state[1][k_pos[0] - 1, k_pos[1]] == 'b' or (k_pos[0] - 1, k_pos[1]) in self.all_camps):
                if action[1] == (k_pos[0] + 1, k_pos[1]) or action[1] == (k_pos[0] - 1, k_pos[1]):
                    return True
            if (state[1][k_pos[0], k_pos[1] - 1] == 'b' or (k_pos[0], k_pos[1] - 1) in self.all_camps) and (state[1][k_pos[0], k_pos[1] + 1] == 'b' or (k_pos[0], k_pos[1] + 1) in self.all_camps):
                if action[1] == (k_pos[0], k_pos[1] + 1) or action[1] == (k_pos[0], k_pos[1] - 1):
                    return True
        if self.draw(state):
            return True
        return False

    # preso lo state ritorna quale gocatore deve muovere (questa info deve essere dentro lo state)
    def to_move(self, state):
        return state[0]

    # definisce la condizione di pareggio (da decidere se implementare)
    def draw(self, state):
        current_checkers = (len(self.white), len(self.black))
        if current_checkers == self.n_checkers:
            if state[1] in self.previous_states:
                return True
        return False

    # presa una condizione di vittoria e un giocatore ritorna un punteggio (?)
    def utility(self, state, player):
        if self.draw(state):
            return 0
        elif state[0] == 'B' and player == 'W':
            return 1
        elif state[0] == 'W' and player == 'W':
            return -1
        elif state[0] == 'W' and player == 'B':
            return 1
        elif state[0] == 'B' and player == 'B':
            return -1

    # trasforma il result in una striga json da mandare al server
    def json(self, state, player):
        pass

    # ritorna true se abbiamo vinto ( while(True): return True)
    def victory(self, state):
        pass

    # res = np.where(array=='k')
    # list(zip(res[0], res[1])) ciao


# Ritorna True se il re ha tutte le linee coperte (da una pedina qualsiasi, da un campo o dal castello
    def free_king_line(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        check = 0
        for dire in self.directions:
            k_pos = (k_pos[0]+dire[0],k_pos[1]+dire[1])
            while state[1][k_pos[0], k_pos[1]] == 'e' or k_pos[0] != 8 or k_pos[0] != 0 or k_pos[1] != 8 or k_pos[1] != 0 or k_pos in self.all_camps or k_pos == self.castle:
                k_pos = (k_pos[0]+dire[0], k_pos[1]+dire[1])
            if not ((k_pos[0] == 0 or k_pos[0] == 8 or k_pos[1] == 0 or k_pos[1] == 8) and state[1][k_pos[0],k_pos[1] == 'e' and k_pos not in self.all_camps]):  # metto questo if perchè io potrei essere uscito nell'ultima cella sia perchè ho trovato una pedina\campo sia perche ho finito la scacchiera
                check += 1
            else:
                check = 0
        return check


    # Ritorna True se tutte le righe e colonne della scacchiera sono coperte
    def free_line(self, state):
        check_r = 0
        check_c = 0
        count = 0
        for i in range(9):
            for j in range(9):
                if (state[1][i,j] == 'e' or state[1][i,j] == 'k') and (i, j) not in self.all_camps and (i, j) not in self.castle:
                    check_r += 1
                if (state[1][j,i] == 'e' or state[1][i,j] == 'k') and (j, i) not in self.all_camps and (j,  i) not in self.castle:
                    check_c += 1
            if check_r != 9:
                count +=1
            if check_c != 9:
                count +=1
        return count


    # Ritorna il numero di pedine nere attaccate al re
    def near_king(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        n_black=0
        for dire in self.directions:
            if state[1][k_pos[0]+dire[0], k_pos[1]+dire[1]] == 'b':
                n_black += 1
        return n_black


    # Ritorna il numero di pedine nere in diagonale
    def diag(self, state):
        black = np.where(state[1] == 'b')
        black = tuple(zip(black[0], black[1]))
        count = 0
        for i in range(len(black)-1):
            for j in range(i+1, len((black)-1)):
                if (black[i][0] == black[j][0] + 1 or black[i][0] == black[j][0] - 1) and (black[i][1] == black[j][1] + 1 or black[i][1] == black[j][1] - 1):
                    count += 1
        return count


    # Ritorna il numero di pedine nere
    def black_pawns(self, state):
        black = np.where(state[1] == 'b')
        black = tuple(zip(black[0], black[1]))
        n_black = len(black)
        return n_black


    # Ritorna il numero di pedine bianche
    def white_pawns(self, state):
        white = np.where(state[1] == 'w')
        white = tuple(zip(white[0], white[1]))
        n_white = len(white)
        return n_white


    # Ritorna True se il re è minacciato
    def king_threat(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        for dire in self.directions:
            new_pos = (k_pos[0] + dire[0], k_pos[1] + dire[1])
            if state[1][new_pos] == 'b' or new_pos in self.all_camps:
                while state[1][new_pos] == 'e' or new_pos not in self.castle or new_pos not in self.all_camps or new_pos[0] != 8 or new_pos[0] != 0 or new_pos[1] != 0 or new_pos[1] != 8:
                    new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                    if state[1][new_pos] == 'b':
                        return True
        return False


    # Return the number of white checkers under threat
    def white_threat(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        count = 0
        for white in white_pos:
            for dire in self.directions:
                new_pos = (white[0] + dire[0], white[1] + dire[1])
                if state[1][new_pos] == 'b' or new_pos in self.all_camps:
                    while state[1][new_pos] == 'e' or new_pos not in self.castle or new_pos not in self.all_camps or new_pos[0] != 8 or new_pos[0] != 0 or new_pos[1] != 0 or new_pos[1] != 8:
                        new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                        if state[1][new_pos] == 'b':
                            count += 1
        return count


    # Return the number of black under threat
    def black_threat(self, state):
        black_pos = np.where(state[1] == 'b')
        black_pos = tuple(zip(black_pos[0], black_pos[1]))
        count = 0
        for black in black_pos:
            for dire in self.directions:
                new_pos = (black[0] + dire[0], black[1] + dire[1])
                if state[1][new_pos] == 'w' or state[1][new_pos] == 'k' or new_pos in self.all_camps:
                    while state[1][new_pos] == 'e' or new_pos not in self.castle or new_pos not in self.all_camps or new_pos[0] != 8 or new_pos[0] != 0 or new_pos[1] != 0 or new_pos[1] != 8:
                        new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                        if state[1][new_pos] == 'w' or state[1][new_pos] == 'k':
                            count += 1
        return count


    def king_outside_castle(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        if k_pos != self.castle:
            return True
        return False


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
        return n_black


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


    def n_white_in_angle(self, state):
        n_white = 0
        if state[1][0,0] == 'w':
            n_white += 1
        if state[1][8,0] == 'w':
            n_white += 1
        if state[1][0,8] == 'w':
            n_white += 1
        if state[1][8,8] == 'w':
            n_white += 1
        return n_white


    def n_white_in_victory(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        n_white = 0
        for white in white_pos:
            if white in self.winning:
                n_white += 1
        return n_white


    def n_white_in_victory_near_white(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        n_white = 0
        for white in white_pos:
            if white in self.winning:
                if ((white[0] + 2 <= 8 and state[1][white[0] + 2, white[1]] == 'w') or (white[1] + 2 <= 8 and state[1][white[0], white[1] + 2] == 'w') or (white[0] - 2 >= 0 and state[1][white[0] - 2, white[1]] == 'w') or (white[1] - 2 >= 0 and state[1][white[0], white[1] - 2] == 'w')):
                    n_white += 1
        return n_white


    def white_evaluation_function(self, state):
        if state[0] == 'W':
            weights = [10, 2, -10, -0.5, -0.6, 1, -80, -2, 1, 2, -1, -0.5, -1, 3, 1, 2]
        else:
            weights = [-80, -2, 10, 0.5, 0.6, -1, 20, 2, -1, -2, 1, 0.5, 1, -3, -1, -2]
        w1 = self.free_king_line(state) * weights[0]
        w2 = self.free_line(state) * weights[1]
        w3 = self.near_king(state) * weights[2]
        w4 = self.diag(state) * weights[3]
        w5 = self.black_pawns(state) * weights[4]
        w6 = self.white_pawns(state) * weights[5]
        w7 = self.king_threat(state) * weights[6]
        w8 = self.white_threat(state) * weights[7]
        w9 = self.black_threat(state) * weights[8]
        w10 = self.king_outside_castle(state) * weights[9]
        w11 = self.black_pawns_in_king_quad(state) * weights[10]
        w12 = self.black_pawns_in_king_near_quad(state) * weights[11]
        w13 = self.black_pawns_when_king_in_cross(state) * weights[12]
        w14 = self.n_white_in_angle(state) * weights[13]
        w15 = self.n_white_in_victory(state) * weights[14]
        w16 = self.n_white_in_victory_near_white(state) * weights[15]
        weights_white = [w1 ,w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16]
        return sum(weights_white)

'''self.initial_state = ('W', np.array([['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                             ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                             ['b', 'b', 'w', 'w', 'k', 'w', 'w', 'b', 'b'],
                                             ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                             ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e']]))

        self.keyboard = np.array([['e', 'w', 'w', 'c', 'c', 'c', 'w', 'w', 'e'],
                                  ['w', 'e', 'e', 'e', 'c', 'e', 'e', 'e', 'e'],
                                  ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                                  ['c', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'c'],
                                  ['c', 'c', 'e', 'e', 'k', 'e', 'e', 'c', 'c'],
                                  ['c', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'c'],
                                  ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                                  ['w', 'e', 'e', 'e', 'c', 'e', 'e', 'e', 'e'],
                                  ['e', 'w', 'w', 'c', 'c', 'c', 'w', 'w', 'e']])

        self.camps = (
        ((0, 3), (0, 4), (0, 5), (1, 4)), ((3, 0), (4, 0), (5, 0), (4, 1)), ((8, 3), (8, 4), (8, 5), (7, 4)),
        ((3, 8), (4, 8), (5, 8), (4, 7)))

        self.all_camps = self.camps[0] + self.camps[1] + self.camps[2] + self.camps[3]

        self.castle = (4, 4)

        self.winning = (
        (0, 1), (0, 2), (1, 0), (2, 0), (0, 6), (0, 7), (6, 0), (7, 0), (8, 1), (8, 2), (1, 8), (2, 8), (6, 8), (7, 8),
        (8, 6), (8, 7))

        self.dimension = (9, 9)

        self.directions = ((0, 1), (0, -1), (1, 0), (-1, 0))  # su giù destra sinistra

        self.central_cross = ((4, 4), (4, 5), (5, 4), (4, 3), (3, 4))

        self.n_checkers = (8,16)

        self.white = ((2, 4), (3, 4), (4, 2), (4, 3), (4, 5), (4, 6), (5, 4), (6, 4), (4, 4))

        self.black = ((0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (3, 8), (4, 0), (4, 1), (4, 7), (4, 8), (5, 0), (5, 8), (7, 4), (8, 3), (8, 4), (8, 5))

        self.previous_states = []  #maybe we will initialize with initial state (depending when we call draw)'''
