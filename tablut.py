from games import Game
import numpy as np

# stato= numpy array riempito con caratteri che rappresentano vuoto, bianco, nero, re ('e','w','b','k')
class Tablut(Game):


    # definisco la scacchiera:
    # -dimensioni
    # -campi
    # -castello
    # -punti di vittoria
    # stato
    def __init__(self):
        self.initial_state = ('W',np.array([['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e'],
                                            ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                            ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                            ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                            ['b', 'b', 'w', 'w', 'k', 'w', 'w', 'b', 'b'],
                                            ['b', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'b'],
                                            ['e', 'e', 'e', 'e', 'w', 'e', 'e', 'e', 'e'],
                                            ['e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e'],
                                            ['e', 'e', 'e', 'b', 'b', 'b', 'e', 'e', 'e']]))

        self.keyboard =      np.array([['e', 'w', 'w', 'c', 'c', 'c', 'w', 'w', 'e'],
                                       ['w', 'e', 'e', 'e', 'c', 'e', 'e', 'e', 'e'],
                                       ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                                       ['c', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'c'],
                                       ['c', 'c', 'e', 'e', 'k', 'e', 'e', 'c', 'c'],
                                       ['c', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'c'],
                                       ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                                       ['w', 'e', 'e', 'e', 'c', 'e', 'e', 'e', 'e'],
                                       ['e', 'w', 'w', 'c', 'c', 'c', 'w', 'w', 'e']])

        self.camps = (((0,3),(0,4),(0,5),(1,4)),((3,0),(4,0),(5,0),(4,1)),((8,3),(8,4),(8,5),(7,4)),((3,8),(4,8),(5,8),(4,7)))

        self.all_camps = self.camps[0]+self.camps[1]+self.camps[2]+self.camps[3]

        self.castle = (4,4)

        self.winning = ((0,1),(0,2),(1,0),(2,0),(0,6),(0,7),(6,0),(7,0),(8,1),(8,2),(1,8),(2,8),(6,8),(7,8),(8,6),(8,7))

        self.dimension = (9,9)

        self.directions = ((0,1), (0,-1), (1,0), (-1,0)) # su giù destra sinistra


    # definisco le azioni possibili:
    # -struttura ((x,y),('sopra',1),'num2,num3,num4) or dictionary(key=(pos_X,pos_Y),value=[(pos_X_finale,pos_Y_finale),(..)...lista delle possibili posizioni finali..]
    # le azioni possibili le passiamo come un generatore
    def actions(self, state):
        black_pos = tuple(zip(*np.where(state[1] == 'b'))) # all current black positions
        white_pos = tuple(zip(*np.where(state[1] == 'w'))) + tuple(zip(*np.where(state[1] == 'k')))  # all current white positions + king position
        possible_actions = []
        invalid_tiles = white_pos + black_pos + self.castle  # tiles in which we can't go in or pass through apart for the camps

        # we check for which player i have to compute the possible moves
        if state[0] == 'B':  # if it's black's turn we check only the black positions
            pos=black_pos
        else:
            pos=white_pos

        # for every initial position (elements)
        for elements in pos:
            if elements not in self.all_camps:
                invalid_tiles = invalid_tiles + self.all_camps  # add all camps as invalid positions
            else:  # if a checker is initially in a camp we add only the other camps as invalid positions
                for i,camp_i in enumerate(self.camps):
                    if elements in camp_i:
                        for j in range(4):
                            if j != i:
                                invalid_tiles = invalid_tiles + self.camps[j]
                        break  # once we find a checkers in a camp we don't check the other camps for it

            # for every direction
            for dire in self.directions:
                checkers = (elements[0]+dire[0],elements[1]+dire[1])  # we compute the final position (checkers) incrementing of 1 in that direction
                while checkers not in invalid_tiles: # while we don't find an invalid position we go on incrementing
                    possible_actions.append((elements,checkers))
                    checkers = (checkers[0] + dire[0], checkers[1] + dire[1])
        return tuple(possible_actions)





    # definiamo lo stato successivo
    def result(self, state, move):
        if state[0] == 'B':  # swap turn
            state[0] = 'W'
            other, my = 'w', 'b'

        else:
            state[0] = 'B'
            other, my = 'b', 'w'

        state[1][move[0]], state[1][move[1]] = state[1][move[1]], state[1][move[0]]  # swap the current position with an empty tile

        for dire in self.directions:
            neighbor = (move[1][0]+dire[0], move[1][1]+dire[1])
            if state[1][neighbor] == other:
                super_neighbor = (neighbor[0] + dire[0], neighbor[1] + dire[1])
                if state[1][super_neighbor] == my:
                    state[1][neighbor] = 'e'


        return state



    # ritorna True quando la scacchiera è in una condizione di vittoria per uno dei due giocaotri
    def terminal_test(self, state):
        k_pos = np.where(state[1] == 'k')
        k_pos = tuple(zip(k_pos[0], k_pos[1])) # mi restituisce una tupla con dentro una tupla che sono la x e la y del king perchè se non metto tuple mi restituirebbe un zip object
        if k_pos[0] in self.winning:
            return True
        if (state[1][k_pos[0]+1,k_pos[1]] == 'b' or (k_pos[0] + 1, k_pos[1]) == self.castle) \
                and (state[1][k_pos[0][0]-1, k_pos[0][1]] == 'b' or (k_pos[0][0]+1, k_pos[0][1]) == self.castle) \
                and (state[1][k_pos[0][0], k_pos[0][1]-1] == 'b' or (k_pos[0][0]+1, k_pos[0][1]) == self.castle) \
                and (state[1][k_pos[0][0], k_pos[0][1]+1] == 'b' or (k_pos[0][0]+1, k_pos[0][1]) == self.castle):
            return True
        return False

    # preso lo state ritorna quale gocatore deve muovere (questa info deve essere dentro lo state)
    def to_move(self, state):
        return state[0]
    # definisce la condizione di pareggio (da decidere se implementare)
    def draw(self, state):
        pass

    # presa una condizione di vittoria e un giocatore ritorna un punteggio (?)
    def utility(self, state, player):
        if self.terminal_test(state) and state[0] == 'B' and player == 'W':
            return 1
        elif self.terminal_test(state) and state[0] == 'W' and player == 'W':
            return -1
        elif self.terminal_test(state) and state[0] == 'W' and player == 'B':
            return 1
        elif self.terminal_test(state) and state[0] == 'B' and player == 'B':
            return -1

    # trasforma il result in una striga json da mandare al server
    def json(self, state, player):
        pass

    # ritorna true se abbiamo vinto ( while(True): return True)
    def victory(self, state):
        pass



    # res = np.where(array=='k')
    # list(zip(res[0], res[1])) ciao