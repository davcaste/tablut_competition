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

        self.camps = ((0,3),(0,4),(0,5),(3,0),(4,0),(5,0),(8,3),(8,4),(8,5),(3,8),(4,8),(5,8))

        self.castle = (4,4)

        self.winning = ((0,1),(0,2),(1,0),(2,0),(0,6),(0,7),(6,0),(7,0),(8,1),(8,2),(1,8),(2,8),(6,8),(7,8),(8,6),(8,7))

        self.dimension = (9,9)


    # definisco le azioni possibili:
    # -struttura ((x,y),('sopra',1),'num2,num3,num4) or dictionary(key=(pos_X,pos_Y),value=[(pos_X_finale,pos_Y_finale),(..)...lista delle possibili posizioni finali..]
    # le azioni possibili le passiamo come un generatore
    def actions(self, state):
        if state[0] == 'B':
            pass


    # definiamo lo stato successivo
    def result(self, state, move):
        pass

    # ritorna True quando la scacchiera è in una condizione di vittoria per uno dei due giocaotri
    def terminal_test(self, state):
        k_pos = np.where(state[1]=='k')
        k_pos = tuple(zip(k_pos[0], k_pos[1])) # mi restituisce una tupla con dentro una tupla che sono la x e la y del king perchè se non metto tuple mi restituirebbe un zip object
        if k_pos[0] in self.winning:
            return True
        if (state[1][k_pos[0]+1,k_pos[1]] == 'b' or (k_pos[0]+1,k_pos[1])==self.castle) \
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
    # list(zip(res[0], res[1]))