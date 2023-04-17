import numpy as np
import os 
import sys, threading
import copy
import random
from copy import deepcopy
from collections import deque

# no draw
# wolf => True, MAXIMIZE player
# sheep => False, MINIMIZE player
INF = 0x7fffff
RANDOM_MODE = True
# MAX_RECURSION_LIMIT = 10000
WOLF_WIN_SCORE = 1
SHEEP_WIN_SCORE = -1
SHEEP = 1
WOLF = 2
SIZE = 5*5*2
BIT_N = 64
STATE_HASH = [random.randint(1, 1<<BIT_N) for i in range(SIZE)]
# sys.setrecursionlimit(MAX_RECURSION_LIMIT)
# threading.stack_size(1<<27)
### Zobrist hashing
# [5][5][2] => 0 for sheep, 1 for wolf
class ZobristHash:
    def __init__(self):
        ''' size => hash table size
            bit_n => n-bit hashing space'''
        # self.state_hash = [random.randint(1, 1<<bit_n) for i in range(size)] # 0 is preserved
        self.score_hash = {}
    
    @staticmethod
    def compress_5_5_2(a, b, c):
        ret = a*10 + b*2 + c*1 
        assert (ret >= 0 and ret < 50)
        return ret
    
    @staticmethod
    def inflate_5_5_2(x):
        a = x / 10
        b = (x%10) / 2
        c = x % 2
        return a, b, c
    
    @staticmethod
    def rehash(hash_key, move, move_made, eat_sheep):
        i,j,k,l = move[0],move[1],move[2],move[3]
        if eat_sheep:
            assert(move_made)
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(i,j,WOLF-1)]
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(k,l,SHEEP-1)]
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(k,l,WOLF-1)]
        elif move_made:
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(i,j,WOLF-1)]
            # hash_val ^= self.state_hash[ZobristHash.compress_5_5_2(k,l,SHEEP-1)]
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(k,l,WOLF-1)] 
        else:
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(i,j,SHEEP-1)]
            # hash_val ^= self.state_hash[ZobristHash.compress_5_5_2(k,l,SHEEP-1)]
            hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(k,l,SHEEP-1)]
        return hash_key
    
    @staticmethod
    def hash(board):
        hash_key = 0
        for i in range(0, 5):
            for j in range(0, 5):
                if board[i][j] == WOLF:
                    hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(i,j,WOLF-1)]
                elif board[i][j] == SHEEP:
                    hash_key ^= STATE_HASH[ZobristHash.compress_5_5_2(i,j,SHEEP-1)]
        return hash_key
    
    def is_in_hashtable(self, key):
        return key in self.score_hash
    
    def get_score_hash_val(self, key):
        return self.score_hash[key]
    
    def add_entry(self, key, val):
        self.score_hash[key] = val

# Tree node
class Node:
    def __init__(self, board, move=None, parent=None, score=0, flag=False):
        '''By default, score = 0 (unknown)'''
        self.board = board
        self.move = move
        self.parent = parent
        self.score = score
        self.flag = flag
        # self.children = children

    def get_children(self, maximize_player):
        current_state = deepcopy(self.board)
        available_moves = []
        children = []
        if maximize_player:
            available_moves = next_move_wolf(current_state)
        else:
            available_moves = next_move_sheep(current_state)
        for move in available_moves:
            tmp_state = deepcopy(current_state)
            eat_sheep = Checker.make_a_move(tmp_state, move, player = 2 if maximize_player else 1)
            children.append(Node(board=tmp_state, move=move, flag=eat_sheep))
            # children.append(Node(board=tmp_state, move=move, parent=self))
        return children
    
    def get_board(self):
        return self.board
    
    def get_move(self):
        return self.move
    
    def get_flag(self):
        return self.flag
    
    def get_score(self):
        return self.score
    
    def set_score(self, score):
        # hash score (?)
        assert(score == 1 or score == -1)
        self.score = score
    
    def get_parent(self):
        return self.parent
    
    def set_parent(self, parent):
        self.parent = parent

    
# game checker    
class Checker:
    def __init__(self) -> None:
        self.hash_table = ZobristHash() # 64-bit zobrist hash for 5*5*2

    @staticmethod
    def make_a_move(state, move, player):
        '''state: a 5*5 matrix
           move: [i, j, k, l] 
           player: 1 / 2'''
        i, j, k, l = move[0], move[1], move[2], move[3]
        flag = True if state[i][j]==2 and state[k][l]==1 else False
        state[i][j] = 0
        state[k][l] = player
        return flag

    def heuristic(self, state, hash_key):
        '''calculate heuristic score with hashing'''
        def count_sheep(matrix):
            cnt = 0
            for i in range(0, 5):
                for j in range(0, 5):
                    if matrix[i][j] == 1:
                        cnt += 1
            return cnt
        ret = (0, None)
        # answer cached
        if self.hash_table.is_in_hashtable(hash_key):
            return self.hash_table.get_score_hash_val(hash_key)
        # sheep wins
        if len(next_move_wolf(state)) == 0:
            ret = (SHEEP_WIN_SCORE, None)
            # self.hash_table.add_entry(key=self.hash_table.hash(state), val=ret)
            # return ret
        # wolf wins
        if count_sheep(state) <= 2:
            ret = (WOLF_WIN_SCORE, None)
            # self.hash_table.add_entry(key=self.hash_table.hash(state), val=ret)
            # return ret
        # hash leaf node
        self.hash_table.add_entry(key=hash_key, val=ret)
        return ret 
    
    # def minimax(self, board, alpha, beta, depth, maximize_player, hash_key):
    #     '''return score, best move'''
    #     # hash_key = ZobristHash.hash(board)
    #     # if repeated 
    #     if self.hash_table.is_in_hashtable(hash_key):
    #         return self.hash_table.get_score_hash_val(hash_key)
    #     print(depth, hash_key)
    #     score = GLOB_GAME_CHECKER.heuristic(board, hash_key=hash_key) 
    #     # leaf node or repeat node
    #     if score != 0:
    #         return score, None
    #     current_state = Node(board=deepcopy(board))
    #     best_move= None
    #     if maximize_player:
    #         max_eval = -INF
    #         for child in current_state.get_children(maximize_player=maximize_player):
    #             child_move, child_eat_sheep = child.get_move(), child.get_flag()
    #             child_hash_key = ZobristHash.rehash(hash_key=hash_key, move=child_move, move_made=maximize_player, eat_sheep=child_eat_sheep)
    #             eval, _ = GLOB_GAME_CHECKER.minimax(child.get_board(), alpha, beta, depth+1, not maximize_player, child_hash_key)
    #             # max_eval = max(max_eval, eval)
    #             if eval > max_eval:
    #                 max_eval = eval
    #                 best_move = child_move
    #             alpha = max(alpha, eval)
    #             if beta <= alpha:
    #                 break
    #         # TODO: hash here
    #         assert(max_eval==WOLF_WIN_SCORE or max_eval==SHEEP_WIN_SCORE)
    #         self.hash_table.add_entry(key=hash_key, val=max_eval)
    #         current_state.set_score(max_eval)
    #         return max_eval, best_move
    #     else:
    #         min_eval = INF
    #         for child in current_state.get_children(maximize_player=maximize_player):
    #             child_move, child_eat_sheep = child.get_move(), child.get_flag()
    #             child_hash_key = ZobristHash.rehash(hash_key=hash_key, move=child_move, move_made=maximize_player, eat_sheep=child_eat_sheep)
    #             eval, _ = GLOB_GAME_CHECKER.minimax(child.get_board(), alpha, beta, depth+1, not maximize_player, child_hash_key)
    #             # min_eval = min(min_eval, eval)
    #             if eval < min_eval:
    #                 min_eval = eval
    #                 best_move = child_move
    #             beta = min(beta, eval)
    #             if beta <= alpha:
    #                 break
    #         # TODO: hash here
    #         assert(min_eval==WOLF_WIN_SCORE or min_eval==SHEEP_WIN_SCORE)
    #         self.hash_table.add_entry(key=hash_key, val=min_eval)
    #         current_state.set_score(min_eval)
    #         return min_eval, best_move
    
    def non_recursive_minimax(self, board, hash_key, maximize_player, depth=0, alpha=-INF, beta=INF):
        # cached answer
        if self.hash_table.is_in_hashtable(hash_key):
            return self.hash_table.get_score_hash_val(hash_key)
        # heuristic value
        (score, move) = GLOB_GAME_CHECKER.heuristic(board, hash_key=hash_key) 
        # leaf node
        if score != 0:
            return score, move
        # explore
        current_state, best_move = Node(board=deepcopy(board)), None  
        stk = []
        stk.append([board, hash_key, maximize_player, depth, alpha, beta])
        while len(stk) > 0:
            [board_, hash_key_, maximize_player_, depth_, alpha_, beta_] = stk.pop()
            if maximize_player_:
                
                for child in current_state.get_children(maximize_player=maximize_player_):
                    child_move, child_eat_sheep = child.get_move(), child.get_flag() 
                    child_hash_key = ZobristHash.rehash(hash_key=hash_key_, move=child_move, move_made=maximize_player_, eat_sheep=child_eat_sheep)

            else:
                pass



        
def load_matrix(matrix_file_name): # read and load the current state
    with open(matrix_file_name, 'r') as f:
        data = f.read()
        data2=data.replace('\n',',').split(',')
    matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            matrix[i,j]=int(data2[5*i+j])
    return matrix

def write_matrix(matrix, matrix_file_name_output): # wirte the new state into new txt file
    with open(matrix_file_name_output, 'w') as f:
        for i in range(5):
            for j in range(5):
                f.write(str(int(matrix[i,j])))
                if j<4:
                    f.write(',')
                if j==4:
                    f.write('\n')

def next_move_wolf(matrix): # random walk for wolf
    candidates=[]
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==2:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        candidates.append([i,j,i+1,j])
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        candidates.append([i,j,i-1,j])
                if j+1<5:
                    if matrix[i,j+1]==0:
                        candidates.append([i,j,i,j+1])
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        candidates.append([i,j,i,j-1])
                # eat sheep
                if i+2<5:
                    if matrix[i+2,j]==1 and matrix[i+1,j]==0:
                        candidates.append([i,j,i+2,j])
                if i-2>=0:
                    if matrix[i-2,j]==1 and matrix[i-1,j]==0:
                        candidates.append([i,j,i-2,j])
                if j+2<5:
                    if matrix[i,j+2]==1 and matrix[i,j+1]==0:
                        candidates.append([i,j,i,j+2])
                if j-2>=0:
                    if matrix[i,j-2]==1 and matrix[i,j-1]==0:
                        candidates.append([i,j,i,j-2])
    # move_idx=np.random.randint(0, len(candidates))
    # return candidates[move_idx]
    return candidates

def next_move_sheep(matrix): # random walk for sheep
    candidates=[]
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==1:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        candidates.append([i,j,i+1,j])
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        candidates.append([i,j,i-1,j])
                if j+1<5:
                    if matrix[i,j+1]==0:
                        candidates.append([i,j,i,j+1])
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        candidates.append([i,j,i,j-1])
    # move_idx=np.random.randint(0, len(candidates))
    # return candidates[move_idx]
    return candidates

GLOB_GAME_CHECKER = Checker()

def AIAlgorithm(filename, movemade): # a showcase for random walk
    iter_num=filename.split('/')[-1]
    iter_num=iter_num.split('.')[0]
    iter_num=int(iter_num.split('_')[1])
    matrix=load_matrix(filename)

    hash_key = ZobristHash.hash(board=matrix)
    # search 
    eval, best_move=GLOB_GAME_CHECKER.minimax(board=matrix, alpha=-INF, beta=INF, depth=0, maximize_player=movemade, hash_key=hash_key)
    print("evaluate:{}".format(eval))
    print(best_move)
    # [start_row, start_col, end_row, end_col] = best_move 
    if movemade==True:
        if RANDOM_MODE:
            candidates = next_move_wolf(matrix) 
            [start_row, start_col, end_row, end_col]=candidates[np.random.randint(0, len(candidates))]
        # move
        matrix2=copy.deepcopy(matrix)
        matrix2[end_row, end_col]=2
        matrix2[start_row, start_col]=0
            
    if movemade==False:
        if RANDOM_MODE:
            candidates = next_move_sheep(matrix) 
            [start_row, start_col, end_row, end_col]=candidates[np.random.randint(0, len(candidates))]
        # move
        matrix2=copy.deepcopy(matrix)
        matrix2[end_row, end_col]=1
        matrix2[start_row, start_col]=0
        
    matrix_file_name_output=filename.replace('state_'+str(iter_num), 'state_'+str(iter_num+1)) 
    write_matrix(matrix2, matrix_file_name_output)

    return start_row, start_col, end_row, end_col