'''TCP Client'''

import games
import tablut
import socket
import json
import numpy as np
import sys

def main():
    if len(sys.argv) != 3:
        exit(1)

    host = sys.argv[1]

    if sys.argv[2] == 5800:     # WIHITE port
        color = 'W'
        port = 5800
    elif sys.argv[2] == 5801:   # BLACK port
        color = 'B'
        port = 5801
    else:
        exit(1)

    client = Client(host, port)
    my_heuristic = tablut.white_evaluation_function()
    search = games.alphabeta_cutoff_search()

    try:
        # present name
        client.send_name("capitano")
        # wait init state
        state_np, turn = client.recv_state()
        # game loop:
        while True:
            if color == turn:
                move = search(state_np, tablut.Tablut, d=2, cutoff_test=None, eval_fn=my_heuristic)
                if move != None:
                    client.send_move(move)
            state_np, turn = client.recv_state()

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
            "from": chr(97 + move[0][1]) + str(move[0][0]+1),
            "to": chr(97 + move[1][1]) + str(move[1][0]+1)
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
        msg = self.sock.recv(total)

        state_obj = json.loads(msg.decode("UTF-8"))
        board = state_obj["board"]
        turn = 'W' if state_obj["turn"] == "WHITE" else 'B'

        state_obj = list(board)
        state_numpy = np.zeros((9,9), dtype = int)
        for i in range(9):
            for j in range(9):
                if state_obj[i][j] == "BLACK":
                    state_numpy[i,j] = 'b'
                if state_obj[i][j] == "WHITE":
                    state_numpy[i,j] = 'w'
                if state_obj[i][j] == "KING":
                    state_numpy[i,j] = 'k'
                if state_obj[i][j] == "EMPTY" or  state_obj[i][j] == "THRONE":
                    state_numpy[i,j] = 'e'

        return state_numpy, turn

    def close(self):
        self.sock.close()