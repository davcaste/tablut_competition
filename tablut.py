from games import Game, alphabeta_cutoff_search
import numpy as np
import copy
import games


# stato= numpy array riempito con caratteri che rappresentano vuoto, bianco, nero, re ('e','w','b','k')
class Tablut(Game):

    # definisco la scacchiera:
    # -dimensioni
    # -campi
    # -castello
    # -punti di vittoria
    # stato
    def __init__(self, ):
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


        # we check for which player i have to compute the possible moves
        if state[0] == 'B':  # if it's black's turn we check only the black positions
            pos = black_pos
        else:
            pos = white_pos
        print('dentro actions')

        # for every initial position (elements)
        for elements in pos:
            invalid_tiles = white_pos + black_pos + self.castle  # tiles in which we can't go in or pass through apart for the camps
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
                    possible_actions.append((elements, checkers))
                    checkers = (checkers[0] + dire[0], checkers[1] + dire[1])

        print('queste sono le possible action')
        print(possible_actions)
        return tuple(possible_actions)

    # definiamo lo stato successivo
    ####### bisogna modificare il fatto che non mangia se schiaccio un bianco contro il castello e dentro c'è il re o
    ####### se schiaccio un nero contro il campo e dentro il campo c'è un nero
    def result(self, state, move):
        print('dentro a result')
        state1 = copy.deepcopy(state)
        state1 = list(state1)
        print ('lo stato è\n:')
        print(state1)
        if state1[0] == 'B':  # swap turn
            state1[0] = 'W'
            other, my = 'w', 'b'

        else:
            state1[0] = 'B'
            other, my = 'b', 'w'

        state1[1][move[0]], state1[1][move[1]] = state1[1][move[1]], state1[1][
            move[0]]  # swap the current position with an empty tile

        for dire in self.directions:
            neighbor = (move[1][0] + dire[0], move[1][1] + dire[1])
            if 0 <= neighbor[0] + dire[0] <= 8 and 0 <= neighbor[1] + dire[1] <= 8:
                if state1[1][neighbor] == other:
                    super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                    if 0 <= super_neighbor[0] + dire[0] <= 8 and 0 <= super_neighbor[1] + dire[1] <= 8:
                        if state1[1][super_neighbor] == my or super_neighbor in self.all_camps or super_neighbor == self.castle:
                            state1[1][neighbor] = 'e'
        print('il nuovo stato è:\n', state1)
        print('calcolo il terminal test')

        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[
            1]))  # mi restituisce una tupla con dentro una tupla che sono la x e la y del king perchè se non metto tuple mi restituirebbe un zip object
        k_pos = k_pos[0]
        print('king position: ', k_pos)
        near = []
        for dire in self.directions:
            supp = (k_pos[0] + dire[0], k_pos[1] + dire[1])
            near.append(supp)
        if k_pos in self.winning:
            state1[0] = 'WW'
        if k_pos in self.central_cross:
            if ((state[1][k_pos[0] + 1, k_pos[1]] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) and (
                    state[1][k_pos[0] - 1, k_pos[1]] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) and (
                    state[1][k_pos[0], k_pos[1] - 1] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) and (
                    state[1][k_pos[0], k_pos[1] + 1] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) and state[0] == 'W'):
                if move[1] in near:
                    state1[0] = 'BW'
        else:
            if (state[1][k_pos[0] + 1, k_pos[1]] == 'b' or (k_pos[0] + 1, k_pos[1]) in self.all_camps) and (
                    state[1][k_pos[0] - 1, k_pos[1]] == 'b' or (k_pos[0] - 1, k_pos[1]) in self.all_camps):
                if move[1] == (k_pos[0] + 1, k_pos[1]) or move[1] == (k_pos[0] - 1, k_pos[1]):
                    state1[0] = 'BW'
            if (state[1][k_pos[0], k_pos[1] - 1] == 'b' or (k_pos[0], k_pos[1] - 1) in self.all_camps) and (
                    state[1][k_pos[0], k_pos[1] + 1] == 'b' or (k_pos[0], k_pos[1] + 1) in self.all_camps):
                if move[1] == (k_pos[0], k_pos[1] + 1) or move[1] == (k_pos[0], k_pos[1] - 1):
                    state1[0] = 'BW'
        print(state1[0])
        return tuple(state1)

    def result1(self, state, move):
        actual_state = copy.deepcopy(state)
        paw = actual_state[1][move[0], move[1]]
        actual_state[1][move[0], move[1]], actual_state[1][move[2], move[3]] = actual_state[1][move[2], move[3]], \
                                                                               actual_state[1][move[0], move[1]]

        # check king on win
        if paw == 'k' and (move[2], move[3]) in self.winning:
            return 'WW', actual_state[1]

        # check mangiato/end
        for dire in self.directions:
            neighbor = (move[2] + dire[0], move[3] + dire[1])
            if (neighbor[0] and neighbor[1]) in range(9):
            # nero mangia bianco
                if paw == 'b' and actual_state[1][neighbor] == 'w':
                    super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                    if (super_neighbor[0] or super_neighbor[1]) not in range(9):
                        continue
                    if actual_state[1][super_neighbor] == 'b' or super_neighbor in (self.all_camps or self.castle):
                        actual_state[1][neighbor] = 'e'

                # nero mangia k
                elif paw == 'b' and actual_state[1][neighbor] == 'k':
                    super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                    if (super_neighbor[0] or super_neighbor[1]) not in range(9):
                        continue
                    # non mangiato
                    if actual_state[1][super_neighbor] == 'w':
                        continue
                    nears = [(neighbor[0] + direc[0], neighbor[1] + direc[1]) for direc in self.directions]
                    # king in posizione generica
                    if (actual_state[1][super_neighbor] == 'b' or super_neighbor in self.all_camps) and (neighbor not in self.castle and self.castle not in nears):
                        return 'BW', actual_state[1]
                    # king nel castello
                    if neighbor in self.castle:
                        for near in nears:
                            if actual_state[1][near] != 'b':
                                continue
                            return 'BW', actual_state
                    if self.castle in nears:
                        for near in nears:
                            if actual_state[1][near] != 'b' and near not in self.castle:
                                continue
                            return 'BW', actual_state[1]


                # bianco/re mangia nero
                elif paw == ('w' or 'k') and actual_state[1][neighbor] == 'b':
                    super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                    if (super_neighbor[0] and super_neighbor[1]) in range(9):
                        # non mangiato
                        if actual_state[1][super_neighbor] == 'b':
                            continue
                        if actual_state[1][super_neighbor] == ('w' or 'k') or super_neighbor in (self.all_camps or self.castle):
                            actual_state[1][neighbor] = 'e'

        # cambio turno
        if actual_state[0] == 'B':
            return 'W', actual_state[1]
        if actual_state[0] == 'W':
            return 'B', actual_state[1]


    # ritorna True quando la scacchiera è in una condizione di vittoria per uno dei due giocaotri
    def terminal_test(self, state):
        print('cerco se è un terminal state')
        if state[0] == 'BW' or state[0] == 'WW':
            return True
        if self.draw(state):
            return True
        return False

    # preso lo state ritorna quale gocatore deve muovere (questa info deve essere dentro lo state)
    def to_move(self, state):
        print('muove il: ',state[0])
        return state[0]

    # definisce la condizione di pareggio (da decidere se implementare)
    def draw(self, state):
        print('guardo se è un pareggio')
        print('questi sono gli stati in cui sono passato:\n', self.previous_states)
        current_checkers = (len(self.white), len(self.black))
        if current_checkers == self.n_checkers:
            if state[1] in self.previous_states:
                return True
        return False

    # presa una condizione di vittoria e un giocatore ritorna un punteggio (?)
    def utility(self, state, player):
        print('partita finita')
        if self.draw(state):
            return 0
        elif state[0] == 'BW' and player == 'W':
            return 1
        elif state[0] == 'WW' and player == 'W':
            return -1
        elif state[0] == 'BW' and player == 'B':
            return 1
        elif state[0] == 'WW' and player == 'B':
            return -1

    # res = np.where(array=='k')
    # list(zip(res[0], res[1])) ciao


# Ritorna True se il re ha tutte le linee coperte (da una pedina qualsiasi, da un campo o dal castello
    def free_king_line(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        check = 0
        for dire in self.directions:
            if 0 <= k_pos[0]+dire[0] <= 8 and 0 <= k_pos[1] + dire[1] <= 8:
                k_pos = (k_pos[0]+dire[0],k_pos[1]+dire[1])
                while state[1][k_pos] == 'e' and (k_pos[0] <= 8 and k_pos[0] >= 0) and (k_pos[1] <= 8 and k_pos[1] >= 0) and k_pos not in self.all_camps and k_pos != self.castle:
                    k_pos = (k_pos[0]+dire[0], k_pos[1]+dire[1])
                if (k_pos[0] == 0 or k_pos[0] == 8 or k_pos[1] == 0 or k_pos[1] == 8) and state[1][k_pos[0], k_pos[1] == 'e' and k_pos not in self.all_camps]:  # metto questo if perchè io potrei essere uscito nell'ultima cella sia perchè ho trovato una pedina\campo sia perche ho finito la scacchiera
                    check += 1
                else:
                    check = 0
        print('controllo linee libere del re: ',check)
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
            if check_r == 9:
                count =1
            if check_c == 9:
                count += 1
        print('controllo linee libere: ', count)
        return count


    # Ritorna il numero di pedine nere attaccate al re
    def near_king(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        n_black=0
        for dire in self.directions:
            if 0 <= k_pos[0] + dire[0] <= 8 and 0 <= k_pos[1] + dire[1] <= 8:
                if state[1][k_pos[0]+dire[0], k_pos[1]+dire[1]] == 'b':
                    n_black += 1
        print('controllo neri vicino al re: ', n_black)
        return n_black


    # Ritorna il numero di pedine nere in diagonale
    def diag(self, state):
        black = np.where(state[1] == 'b')
        black = tuple(zip(black[0], black[1]))
        count = 0
        for i in range((len(black))-1):
            for j in range(i+1, (len(black)) - 1):
                if (black[i][0] == black[j][0] + 1 or black[i][0] == black[j][0] - 1) and (black[i][1] == black[j][1] + 1 or black[i][1] == black[j][1] - 1):
                    count += 1
        print('controllo neri in diagonale: ',count)
        return count


    # Ritorna il numero di pedine nere
    def black_pawns(self, state):
        black = np.where(state[1] == 'b')
        black = tuple(zip(black[0], black[1]))
        n_black = len(black)
        print('controllo numero di pedine nere: ',n_black)
        return n_black


    # Ritorna il numero di pedine bianche
    def white_pawns(self, state):
        white = np.where(state[1] == 'w')
        white = tuple(zip(white[0], white[1]))
        n_white = len(white)
        print('controllo numero di pedine bianche: ',n_white)
        return n_white


    # Ritorna True se il re è minacciato
    def king_threat(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        print('controllo se il re è minacciato:')
        for dire in self.directions:
            if 0 <= k_pos[0] + dire[0] <= 8 and 0 <= k_pos[1] + dire[1] <= 8:
                new_pos = (k_pos[0] + dire[0], k_pos[1] + dire[1])
                if state[1][new_pos] == 'b' or new_pos in self.all_camps:
                    while state[1][new_pos] == 'e' or new_pos not in self.castle or new_pos not in self.all_camps or new_pos[0] != 8 or new_pos[0] != 0 or new_pos[1] != 0 or new_pos[1] != 8:
                        new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                        if state[1][new_pos] == 'b':
                            print('il re è minacciato')
                            return True
        print('il re non è minacciato')
        return False


    # Return the number of white checkers under threat
    def white_threat(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        count = 0
        for white in white_pos:
            for dire in self.directions:
                if 0 <= white[0] + dire[0] <= 8 and 0 <= white[1] + dire[1] <= 8:
                    new_pos = (white[0] + dire[0], white[1] + dire[1])
                    if state[1][new_pos] == 'b' or new_pos in self.all_camps:
                        while state[1][new_pos] == 'e' and new_pos not in self.castle and new_pos not in self.all_camps and (new_pos[0] <= 8 and new_pos[0] >= 0) and (new_pos[1] >= 0 and new_pos[1] <= 8):
                            new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                            if state[1][new_pos] == 'b':
                                count += 1
        print('il numero di bianchi minacciati è: ',count)
        return count


    # Return the number of black under threat
    def black_threat(self, state):
        black_pos = np.where(state[1] == 'b')
        black_pos = tuple(zip(black_pos[0], black_pos[1]))
        count = 0
        print('dentro a black treat')
        for black in black_pos:
            for dire in self.directions:
                if 0 <= black[0] + dire[0] <= 8 and 0 <= black[1] + dire[1] <= 8:
                    new_pos = (black[0] + dire[0], black[1] + dire[1])
                    if new_pos[0] >= 0 and new_pos[0] <= 8 and new_pos[1] <= 8 and new_pos[1] >= 0:
                        while state[1][new_pos] == 'e' and new_pos not in self.castle and new_pos not in self.all_camps and (new_pos[0] < 8 and new_pos[0] > 0) and (new_pos[1] > 0 and new_pos[1] < 8):
                            new_pos = (new_pos[0] + dire[0], new_pos[1] + dire[1])
                            if state[1][new_pos] == 'w' or state[1][new_pos] == 'k':
                                count += 1
        print('il numero di neri minacciati è: ',count)
        return count


    def king_outside_castle(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1]))
        k_pos = k_pos[0]
        if k_pos != self.castle:
            print('il re è fuori dal castello')
            return True
        print('il re è dentro al castello')
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
        print('nuero di neri nel quadrante del re: ',n_black)
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
        print('numero di neri nei quadranti vicino a quello del re: ',n_black)
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
        print('numero di neri nei 2 quadranti vicini al re se il re è nella croce centrale: ', n_black)
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
        print('numero di bianchi negli angoli: ',n_white)
        return n_white


    def n_white_in_victory(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        n_white = 0
        for white in white_pos:
            if white in self.winning:
                n_white += 1
        print('numero di bianchi nelle caselle della vittoria: ',n_white)
        return n_white


    def n_white_in_victory_near_white(self, state):
        white_pos = np.where(state[1] == 'w')
        white_pos = tuple(zip(white_pos[0], white_pos[1]))
        n_white = 0
        for white in white_pos:
            if white in self.winning:
                if ((white[0] + 2 <= 8 and state[1][white[0] + 2, white[1]] == 'w') or (white[1] + 2 <= 8 and state[1][white[0], white[1] + 2] == 'w') or (white[0] - 2 >= 0 and state[1][white[0] - 2, white[1]] == 'w') or (white[1] - 2 >= 0 and state[1][white[0], white[1] - 2] == 'w')):
                    n_white += 1
        print('numero di bianchi nelle caselle della vittoria con anche dei bianchi vicino: ',n_white)
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
print ('inizio')
heur = Tablut().white_evaluation_function
print('ho l euristica')
init_state = ('W', np.array([['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                             ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                             ['b', 'b', 'w', 'w', 'k', 'w', 'w', 'b', 'b'],
                                             ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                             ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                             ['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e']]))
print('inizio la ricerca')
search = games.alphabeta_cutoff_search(init_state, Tablut(), eval_fn=heur)

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
