'''Client'''

import my_games
import tablut
import socket
import json
import numpy as np
import sys
import threading as t
import multiprocessing as mp
import time

move = None
m_value = - float('inf')
stop_flag = False


def main():
    global move, stop_flag      # shared variables

    if len(sys.argv) != 3:
        exit(1)

    host = sys.argv[1]

    if sys.argv[2] == '5800':     # WIHITE port
        color = 'W'
        port = 5800
    elif sys.argv[2] == '5801':   # BLACK port
        color = 'B'
        port = 5801
    else:
        exit(1)

    lock = t.Lock()     # lock needed to acces critical section (global move)

    client = Client(host, port)
    my_heuristic = tablut.Tablut().white_evaluation_function
    search = my_games.alphabeta_cutoff_search   # NB: my_games (not games)

    try:
        # present name
        client.send_name("Capitano")

        # wait init state
        turn, state_np = client.recv_state()
        print(turn, state_np)

        # game loop:
        while True:
            if color == turn:
                # Timer used to not exceed the timeout
                tim = t.Timer(30.0, function=timer, args=[client, lock])
                tim.start()
                
                # MultiProcessing implementation
                processes = [mp.Process(target=actual, args=[lock, i+1, search, turn, state_np, my_heuristic]) for i in range(2)]
                [process.start() for process in processes]

                while not stop_flag:
                    pass

                [process.terminate() for process in processes]
                tim.join()

                # move = search((turn, state_np), tablut.Tablut(), d=1, cutoff_test=None, eval_fn=my_heuristic)

            turn, state_np = client.recv_state()
            print (state_np, turn)

    finally:
        print('closing socket')
        client.close()


class Client:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send_name(self, name):
        encoded = name.encode("UTF-8")
        length = len(encoded).to_bytes(4, 'big')
        self.sock.sendall(length+encoded)

    def send_move(self, move):
        move_obj = {
            "from": chr(97 + move[1]) + str(move[0]+1),
            "to": chr(97 + move[3]) + str(move[2]+1)
        }

        encoded = json.dumps(move_obj).encode("UTF-8")
        length = len(encoded).to_bytes(4, 'big')
        self.sock.sendall(length+encoded)

    def recv_state(self):
        char = self.sock.recv(1)
        while(char == b'\x00'):
            char = self.sock.recv(1)
        length_str = char + self.sock.recv(1)
        total = int.from_bytes(length_str, "big")
        state = self.sock.recv(total).decode("UTF-8")

        state = state.replace('EMPTY', 'e')
        state = state.replace('THRONE', 'e')
        state = state.replace('KING', 'k')
        state = state.replace('BLACK', 'b')
        state = state.replace('WHITE', 'w')

        state_dict = json.loads(state)
        matrix = np.array(state_dict['board'])

        return state_dict['turn'].capitalize(), matrix

    def close(self):
        self.sock.close()


def timer(client, lock):
    '''
    Function used to handle the timing contraints to produce an action
    '''
    global move, stop_flag

    lock.acquire()
    print("----------------------->",move)
    if move != None:
        client.send_move(move)
    lock.release()

    stop_flag = True


def actual(lock, part, search, turn, state_np, my_heuristic):
    '''
    Function used to search in a specific subdomain of possible actions
    '''
    global move, m_value
    for depth in range(1, 10):
        # NB: TWO (not one) VALUES RETURNED FROM SEARCH
        action, our_value = search((turn, state_np), tablut.Tablut(), d=depth, cutoff_test=None, eval_fn=my_heuristic,
                               part=part)
        lock.acquire()
        print ("--------------", our_value, m_value, "--------------")
        if our_value > m_value:
            move = action
            m_value = our_value
        lock.release()


if __name__ == '__main__': main()
