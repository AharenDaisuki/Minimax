import numpy as np
import os 
import copy
from random import randint
from random import shuffle
from time import time

# game setting
INF = 0x7fffff
SHEEP_N = 10
ENABLE_TIMER = False
MAX_REPEAT = 10

# heuristic
TRAPED = -50
EAT_SHEEP = 10
QI = 1
WOLF_WIN = 1000
SHEEP_WIN = -1000

# parameters
WOLF_DEPTH = 1<<3
SHEEP_DEPTH = 1<<3
SEARCH_DEPTH = 1<<3
MAX_TABLE_SIZE = 1<<20 #1024 * 1024

# utils
def load_matrix(matrix_file_name): 
    with open(matrix_file_name, 'r') as f:
        data = f.read()
        data2=data.replace('\n',',').split(',')
    matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            matrix[i,j]=int(data2[5*i+j])
    return matrix

def write_matrix(matrix, matrix_file_name_output): 
    with open(matrix_file_name_output, 'w') as f:
        for i in range(5):
            for j in range(5):
                f.write(str(int(matrix[i,j])))
                if j<4:
                    f.write(',')
                if j==4:
                    f.write('\n')

def all_wolf_moves(state):
    candidates = []
    for i in range(5):
        for j in range(5):
            if AI.is_wolf(state, i, j):
                if not AI.out_of_board(i+1, j) and AI.is_empty(state, i+1, j):
                    candidates.append([i,j,i+1,j])
                if not AI.out_of_board(i-1, j) and AI.is_empty(state, i-1, j):
                    candidates.append([i,j,i-1,j])
                if not AI.out_of_board(i, j+1) and AI.is_empty(state, i, j+1):
                    candidates.append([i,j,i,j+1])
                if not AI.out_of_board(i, j-1) and AI.is_empty(state, i, j-1):
                    candidates.append([i,j,i,j-1])
                if not AI.out_of_board(i+2, j) and AI.is_empty(state, i+1, j) and AI.is_sheep(state, i+2, j):
                    candidates.append([i,j,i+2,j])
                if not AI.out_of_board(i-2, j) and AI.is_empty(state, i-1, j) and AI.is_sheep(state, i-2, j):
                    candidates.append([i,j,i-2,j])
                if not AI.out_of_board(i, j+2) and AI.is_empty(state, i, j+1) and AI.is_sheep(state, i, j+2):
                    candidates.append([i,j,i,j+2])
                if not AI.out_of_board(i, j-2) and AI.is_empty(state, i, j-1) and AI.is_sheep(state, i, j-2):
                    candidates.append([i,j,i,j-2])
    shuffle(candidates)
    return candidates

def all_sheep_moves(state):
    candidates = []
    for i in range(5):
        for j in range(5):
            if AI.is_sheep(state, i, j):
                if not AI.out_of_board(i+1, j) and AI.is_empty(state, i+1, j):
                    candidates.append([i,j,i+1,j])
                if not AI.out_of_board(i-1, j) and AI.is_empty(state, i-1, j):
                    candidates.append([i,j,i-1,j])
                if not AI.out_of_board(i, j+1) and AI.is_empty(state, i, j+1):
                    candidates.append([i,j,i,j+1])
                if not AI.out_of_board(i, j-1) and AI.is_empty(state, i, j-1):
                    candidates.append([i,j,i,j-1])
    shuffle(candidates)
    return candidates

def compress_index(a, b, c):
    return int(a*10 + b*2 + c)

def inflate_index(x):
    return int(x//10), int((x%10)//2), int(x%2)

# timer (only for testing)
class Timer:
    def __init__(self):
        self.time = 0.0
        self.start = None
        self.end = None
    def tic(self):
        if ENABLE_TIMER:
            self.start = time()
    def toc(self):
        if ENABLE_TIMER:
            self.end = time()
            duration = self.end - self.start
            self.time += duration
            return duration
    def get_time(self):
        if ENABLE_TIMER:
            return self.time

# AI
class AI:
    def __init__(self, size, bit_n):
        ''' size: size of state, bit_n: number of hash bit '''
        # initialize zobrist 
        self.zobrist = [randint(1, 1<<bit_n) for i in range(size)]
        self.hash_table = {}

    @staticmethod
    def is_wolf(state, i, j):
        return state[i][j] == 2
    @staticmethod
    def is_sheep(state, i, j):
        return state[i][j] == 1
    @staticmethod
    def is_empty(state, i, j):
        return state[i][j] == 0
    @staticmethod
    def out_of_board(i, j):
        return i<0 or i>=5 or j<0 or j>=5
    @staticmethod
    def count_qi(state, i, j):
        qi = 0
        if not AI.out_of_board(i-1, j):
            if AI.is_empty(state, i-1, j):
                qi += 1
        if not AI.out_of_board(i+1, j):
            if AI.is_empty(state, i+1, j):
                qi += 1
        if not AI.out_of_board(i, j-1):
            if AI.is_empty(state, i, j-1):
                qi += 1
        if not AI.out_of_board(i, j+1):
            if AI.is_empty(state, i, j+1):
                qi += 1
        return qi
    @staticmethod
    def count_sheep(state):
        sheep_n = 0
        for i in range(5):
            for j in range(5):
                if AI.is_sheep(state, i, j):
                    sheep_n += 1
        return sheep_n
    @staticmethod
    def count_alive_wolf(state):
        alive = 0
        qi_total = 0
        for i in range(5):
            for j in range(5):
                if AI.is_wolf(state, i, j):
                    qi = AI.count_qi(state, i, j)
                    if qi > 0:
                        alive += 1
                        qi_total += qi
        return alive, qi_total
    @staticmethod
    def eat_sheep(state):
        eat_sheep_n = 0
        for i in range(5):
            for j in range(5):
                if AI.is_wolf(state, i, j):
                    tmp = 0
                    if tmp<2 and not AI.out_of_board(i+2,j) and AI.is_sheep(i+2,j):
                        tmp += 1
                    if tmp<2 and not AI.out_of_board(i-2,j) and AI.is_sheep(i-2,j):
                        tmp += 1
                    if tmp<2 and not AI.out_of_board(i,j+2) and AI.is_sheep(i,j+2):
                        tmp += 1
                    if tmp<2 and not AI.out_of_board(i,j-2) and AI.is_sheep(i,j-2):
                        tmp += 1 
                    assert(tmp <= 2)
                    eat_sheep_n += tmp
        assert(eat_sheep_n <= 4)
        return eat_sheep_n
    
    @staticmethod
    def checkmate(wolf_turn, state):
        # print('checkmate?') # TODO: remove log
        if wolf_turn:
            # wolf checkmate
            # for move in next_move_wolf(matrix=state):
            for move in all_wolf_moves(state):
                k, l = move[2], move[3]
                if AI.is_sheep(state, k, l):
                    # print('check!')
                    return move
        else:
            # sheep checkmate
            # for move in next_move_sheep(matrix=state):
            for move in all_sheep_moves(state):
                i, j, k, l = move[0], move[1], move[2], move[3]
                # test move
                state[i][j] = 0
                state[k][l] = 1
                # evaluate
                alive_wolf_n, qi_n = AI.count_alive_wolf(state)
                # undo move
                state[i][j] = 1
                state[k][l] = 0
                if alive_wolf_n == 0 and qi_n == 0:
                    # print('check!')
                    return move
        return None

    def hash(self, state):
        '''0 for sheep, 1 for wolf'''
        hash_key = 0
        for i in range(5):
            for j in range(5):
                if AI.is_sheep(state,i,j):
                    hash_key ^= self.zobrist[compress_index(i,j,0)]
                elif AI.is_wolf(state,i,j):
                    hash_key ^= self.zobrist[compress_index(i,j,1)]
        return hash_key

    def register(self, hash_key, hash_val, move, depth):
        if hash_key not in self.hash_table:
            self.hash_table[hash_key] = {}
        self.hash_table[hash_key]['score'] = hash_val
        self.hash_table[hash_key]['move'] = move
        self.hash_table[hash_key]['depth'] = depth
        self.hash_table[hash_key]['hit'] = 0

    def heuristic(self, state, wolf_turn):
        def is_wolf(state, i, j):
            return state[i][j] == 2
        def is_sheep(state, i, j):
            return state[i][j] == 1
        def is_empty(state, i, j):
            return state[i][j] == 0
        def out_of_board(i, j):
            return i<0 or i>=5 or j<0 or j>=5
        def count_qi(state, i, j):
            qi = 0
            if not out_of_board(i-1, j):
                if is_empty(state, i-1, j):
                    qi += 1
            if not out_of_board(i+1, j):
                if is_empty(state, i+1, j):
                    qi += 1
            if not out_of_board(i, j-1):
                if is_empty(state, i, j-1):
                    qi += 1
            if not out_of_board(i, j+1):
                if is_empty(state, i, j+1):
                    qi += 1
            return qi
        def count_sheep(state):
            sheep_n = 0
            for i in range(5):
                for j in range(5):
                    if is_sheep(state, i, j):
                        sheep_n += 1
            return sheep_n
        def count_alive_wolf(state):
            alive = 0
            qi_total = 0
            for i in range(5):
                for j in range(5):
                    if is_wolf(state, i, j):
                        qi = count_qi(state, i, j)
                        if qi > 0:
                            alive += 1
                            qi_total += qi
            return alive, qi_total
        def checkmate(wolf_turn, state):
            # print('checkmate?') # TODO: remove log
            if wolf_turn:
                # wolf checkmate
                # for move in next_move_wolf(matrix=state):
                for move in all_wolf_moves(state):
                    k, l = move[2], move[3]
                    if is_sheep(state, k, l):
                        # print('check!')
                        return move
            else:
                # sheep checkmate
                # for move in next_move_sheep(matrix=state):
                for move in all_sheep_moves(state):
                    i, j, k, l = move[0], move[1], move[2], move[3]
                    # test move
                    state[i][j] = 0
                    state[k][l] = 1
                    # evaluate
                    alive_wolf_n, qi_n = count_alive_wolf(state)
                    # undo move
                    state[i][j] = 1
                    state[k][l] = 0
                    if alive_wolf_n == 0 and qi_n == 0:
                        # print('check!')
                        return move
            return None


        ''' give a heuristic value given a state'''
        sheep_n = count_sheep(state)
        # wolf win
        if sheep_n <= 2:
            return WOLF_WIN, None # 1000
        if wolf_turn and sheep_n == 3:
            # wolf checkmate
            if not (kill := checkmate(wolf_turn, state)) == None:
                return WOLF_WIN, kill
        # sheep win
        alive_wolf_n, qi_n = count_alive_wolf(state)
        if alive_wolf_n == 0:
            return SHEEP_WIN, None # -1000
        score = 0
        # traped
        if alive_wolf_n == 1:
            score += TRAPED
            # sheep checkmate
            if not wolf_turn and qi_n == 1:
                if not (kill := checkmate(wolf_turn, state)) == None:
                    return SHEEP_WIN, kill
        # eat sheep
        score += (SHEEP_N - sheep_n) * EAT_SHEEP
        # qi
        score += qi_n * QI
        return score, None
    
    def make_a_move(self, state, hash_key, move):
        [i,j,k,l] = move
        from_, to_ = state[i][j], state[i][j]
        # hash
        flag = False # eat sheep
        hash_key ^= self.zobrist[compress_index(i,j,from_-1)]
        hash_key ^= self.zobrist[compress_index(k,l,from_-1)]
        if AI.is_sheep(state, k, l):
            flag = True
            hash_key ^= self.zobrist[compress_index(k,l,to_-1)]
        # move
        state[i][j] = 0
        state[k][l] = from_
        return hash_key, flag
    
    def undo_a_move(self, state, move, eat_sheep):
        [i,j,k,l] = move
        state[i][j] = state[k][l]
        state[k][l] = 1 if eat_sheep else 0

    def max_move(self, state, hash_key, depth, alpha=-INF, beta=INF):
        # if answer cached
        if hash_key in self.hash_table and self.hash_table[hash_key]['hit'] <= MAX_REPEAT:
            if self.hash_table[hash_key]['depth'] >= depth:
                self.hash_table[hash_key]['hit'] += 1
                return self.hash_table[hash_key]['score'], self.hash_table[hash_key]['move']
        # heuristic 
        h, check = self.heuristic(state, wolf_turn=True)
        if depth == 0 or h == SHEEP_WIN or h == WOLF_WIN:
            return h, check
        # check children
        value, best_move = -INF, None
        # for move in next_move_wolf(matrix=state):
        for move in all_wolf_moves(state):
            child_hash_key, eat_sheep = self.make_a_move(state=state, hash_key=hash_key, move=move) # state modified
            score, _ = self.min_move(state=state, hash_key=child_hash_key, depth=depth-1, alpha=alpha, beta=beta)
            self.undo_a_move(state=state, move=move, eat_sheep=eat_sheep) # undo the move
            # break if win
            if score == WOLF_WIN:
                self.register(hash_key=hash_key, hash_val=WOLF_WIN, move=move, depth=depth)
                return WOLF_WIN, move
            if score > value:
                value, best_move = score, move
                alpha = max(alpha, value)
            if value > beta:
                # break
                self.register(hash_key=hash_key, hash_val=value, move=best_move, depth=depth)
                return value, best_move
            if value == beta:
                # break for p=0.5
                if randint(0, 1):
                    self.register(hash_key=hash_key, hash_val=value, move=best_move, depth=depth)
                    return value, best_move
        self.register(hash_key=hash_key, hash_val=value, move=best_move, depth=depth)
        return value, best_move


    def min_move(self, state, hash_key, depth, alpha=-INF, beta=INF):
        # if answer cached
        if hash_key in self.hash_table and self.hash_table[hash_key]['hit'] <= MAX_REPEAT:
            if self.hash_table[hash_key]['depth'] >= depth:
                self.hash_table[hash_key]['hit'] += 1    
                return self.hash_table[hash_key]['score'], self.hash_table[hash_key]['move']
        # heuristic 
        h, check = self.heuristic(state, wolf_turn=False)
        if depth == 0 or h == SHEEP_WIN or h == WOLF_WIN:
            return h, check
        # check children
        value, best_move = INF, None
        # for move in next_move_sheep(matrix=state):
        for move in all_sheep_moves(state):
            child_hash_key, eat_sheep = self.make_a_move(state=state, hash_key=hash_key, move=move) # state modified
            score, _ = self.max_move(state=state, hash_key=child_hash_key, depth=depth-1, alpha=alpha, beta=beta)
            self.undo_a_move(state=state, move=move, eat_sheep=eat_sheep) # undo the move
            if score == SHEEP_WIN:
                self.register(hash_key=hash_key, hash_val=SHEEP_WIN, move=move, depth=depth)
                return SHEEP_WIN, move 
            if score < value:
                value, best_move = score, move
                beta = min(beta, value)
            if value < alpha:
                # break
                self.register(hash_key=hash_key, hash_val=value, move=best_move, depth=depth)
                return value, best_move
            if value == alpha:
                # break for p=0.5
                if randint(0, 1):
                    self.register(hash_key=hash_key, hash_val=value, move=best_move, depth=depth)
                    return value, best_move
        self.register(hash_key=hash_key, hash_val=value, move=best_move, depth=depth)
        return value, best_move 
    
    def minimax(self, state, wolf_turn, depth=(WOLF_DEPTH, SHEEP_DEPTH)):
        # checkmate
        if wolf_turn and self.count_sheep(state)==3:
            # print('checkmate?')
            if not (kill := AI.checkmate(wolf_turn=wolf_turn, state=state)) == None:
                # print('check!')
                return WOLF_WIN, kill
        # if not wolf_turn and len(next_move_wolf(state))==1:
        if not wolf_turn and len(all_wolf_moves(state)) == 1:
            # print('checkmate?')
            if not (kill := AI.checkmate(wolf_turn=wolf_turn, state=state)) == None:
                # print('check!')
                return SHEEP_WIN, kill
        if len(self.hash_table) > MAX_TABLE_SIZE:
            self.hash_table = {}
        hash_key = self.hash(state=state)
        if wolf_turn:
            return self.max_move(state=state, hash_key=hash_key, depth=depth[0])
        else:
            return self.min_move(state=state, hash_key=hash_key, depth=depth[1])

GLOB_GAME_CHECKER = AI(size=5*5*2, bit_n=64) # state space: 50, bit: 64
GLOB_TIMER = Timer() # global timer

def AIAlgorithm(filename, movemade): # a showcase for random walk
    iter_num=filename.split('/')[-1]
    iter_num=iter_num.split('.')[0]
    iter_num=int(iter_num.split('_')[1])
    matrix=load_matrix(filename)

    GLOB_TIMER.tic()
    score, move = GLOB_GAME_CHECKER.minimax(state=matrix, wolf_turn=movemade)
    duration = GLOB_TIMER.toc()
    if ENABLE_TIMER:
        print('[{}] score: {}, move: {}, cache: {}, time:{}, cumulative time: {}'.format(iter_num+1, score, move, len(GLOB_GAME_CHECKER.hash_table), duration, GLOB_TIMER.get_time()))
    [start_row, start_col, end_row, end_col] = move
    matrix2=copy.deepcopy(matrix) 
    if movemade==True:   
        matrix2[end_row, end_col]=2
        matrix2[start_row, start_col]=0
            
    if movemade==False:
        matrix2[end_row, end_col]=1
        matrix2[start_row, start_col]=0
        
    matrix_file_name_output=filename.replace('state_'+str(iter_num), 'state_'+str(iter_num+1)) 
    write_matrix(matrix2, matrix_file_name_output)

    return start_row, start_col, end_row, end_col