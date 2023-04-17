import numpy as np
import os 
import copy
import random
from ai_algorithm_56638660 import Timer
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

def sortedDictValues(my_dict):
    v = max(my_dict.values())
    res=[]

    for key in my_dict.keys():
        if my_dict[key]==v:
            res.append(key)
    #print(self.my_dict)
    if len(res)==1:
        # print(res)
        return list(res[0])
    else:
        # print(res)
        return list(random.choice(res))

def next_move_wolf(matrix): # random walk for wolf
    resultdict=score_wolf_possible_actions(matrix=matrix)

    return sortedDictValues(resultdict)

def next_move_sheep(matrix): # random walk for sheep
    resultdict=score_sheep(matrix=matrix)

    return sortedDictValues(resultdict)

def getallpossibleactions_wolf(matrix): # random walk for wolf
    candidates=[]
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==2:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        candidates.append(tuple([i,j,i+1,j]))
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        candidates.append(tuple([i,j,i-1,j]))
                if j+1<5:
                    if matrix[i,j+1]==0:
                        candidates.append(tuple([i,j,i,j+1]))
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        candidates.append(tuple([i,j,i,j-1]))
                if i+2<5:
                    if matrix[i+2,j]==1 and matrix[i+1,j]==0:
                        candidates.append(tuple([i,j,i+2,j]))
                if i-2>=0:
                    if matrix[i-2,j]==1 and matrix[i-1,j]==0:
                        candidates.append(tuple([i,j,i-2,j]))
                if j+2<5:
                    if matrix[i,j+2]==1 and matrix[i,j+1]==0:
                        candidates.append(tuple([i,j,i,j+2]))
                if j-2>=0:
                    if matrix[i,j-2]==1 and matrix[i,j-1]==0:
                        candidates.append(tuple([i,j,i,j-2]))
    #move_idx=np.random.randint(0, len(candidates))
    return candidates

def check_how_many_empty_pos_around_wolf_corner(matrix,action):
    i=action[2]
    j=action[3]
    count=0
    if (i-1<0 and j-1<0):       
        if matrix[i+1,j]==1 and matrix[i,j+1]==0:
            if matrix[i,j+2]==1:
                count-=800
        if matrix[i,j+1]==1 and matrix[i+1,j]==0:
            if matrix[i+2,j]==1:
                count-=800
    if i-1<0 and j+1==5:
        if matrix[i+1,j]==1 and matrix[i,j-1]==0:
            if matrix[i,j-2]==1:
                count-=800
        if matrix[i,j-1]==1 and matrix[i+1,j]==0:
            if matrix[i+2,j]==1:
                count-=800
    if i==4 and j==0:
        if matrix[i-1,j]==1 and matrix[i,j+1]==0:
            if matrix[i,j+2]==1:
                count-=800
        if matrix[i,j+1]==1 and matrix[i-1,j]==0:
            if matrix[i-2,j]==1:
                count-=800
    if i==4 and j==4:
        if matrix[i-1,j]==1 and matrix[i,j-1]==0:
            if matrix[i,j-2]==1:
                count-=800
        if matrix[i,j-1]==1 and matrix[i-1,j]==0:
            if matrix[i-2,j]==1:
                count-=800
    return count

def check_how_many_empty_pos_around_wolf_line(matrix,action):
    i=action[2]
    j=action[3]
    count=0
    if action[2]==4:
        if matrix[i,j-1]==1 and matrix[i-1,j]==1:
            if j+2>4 and matrix[i-2,j+1]==1:
                count-=500
            if j+2<=4 and matrix[i,j+2]==1:
                count-=500
        if matrix[i,j-1]==1 and matrix[i,j+1]==1:
            if matrix[i-2,j]==1:
                count-=500
        if matrix[i-1,j]==1 and matrix[i,j+1]==1:
            if j-2<0 and matrix[i-2,j-1]==1:
                count-=500
            if j-2>=0 and matrix[i,j-2]==1:
                count-=500
    if action[2]==0:
        if matrix[i,j-1]==1 and matrix[i+1,j]==1:
            if j+2>4 and matrix[i+2,j+1]==1:
                count-=500
            if j+2<=4 and matrix[i,j+2]==1:
                count-=500
        if matrix[i,j-1]==1 and matrix[i,j+1]==1:
            if matrix[i+2,j]==1:
                count-=500
        if matrix[i+1,j]==1 and matrix[i,j+1]==1:
            if j-2<0 and matrix[i+2,j-1]==1:
                count-=500
            if j-2>=0 and matrix[i,j-2]==1:
                count-=500
    if action[3]==0:
        if matrix[i-1,j]==1 and matrix[i,j+1]==1:
            if i+2>4 and matrix[i+1,j+2]==1:
                count-=500
            if i+2<=4 and matrix[i+2,j]==1:
                count-=500
        if matrix[i-1,j]==1 and matrix[i+1,j]==1:
            if matrix[i,j+2]==1:
                count-=500
        if matrix[i,j+1]==1 and matrix[i-1,j]==1:
            if i-2<0 and matrix[i-1,j+2]==1:
                count-=500
            if i-2>=0 and matrix[i-2,j]==1:
                count-=500
    if action[3]==4:
        if matrix[i-1,j]==1 and matrix[i,j-1]==1:
            if i+2>4 and matrix[i+1,j-2]==1:
                count-=500
            if i+2<=4 and matrix[i+2,j]==1:
                count-=500
        if matrix[i-1,j]==1 and matrix[i+1,j]==1:
            if matrix[i,j-2]==1:
                count-=500
        if matrix[i,j-1]==1 and matrix[i+1,j]==1:
            if i-2<0 and matrix[i-1,j-2]==1:
                count-=500
            if i-2>=0 and matrix[i-2,j]==1:
                count-=500
    return count

def check_how_many_around_wolf_middle(matrix,action):
    i=action[2]
    j=action[3]
    count=0
    if matrix[i-1,j]==1 and matrix[i,j+1]==1 and matrix[i+1,j]==1:
        if matrix[i-1,j-1]==1 or matrix[i+1,j-1]==1:
            count-=300
    if matrix[i-1,j]==1 and matrix[i+1,j]==1 and matrix[i,j-1]==1:
        if matrix[i-1,j+1]==1 or matrix[i+1,j+1]==1:
            count-=300
    if matrix[i,j+1]==1 and matrix[i+1,j]==1 and matrix[i,j-1]==1:
        if matrix[i-1,j-1]==1 or matrix[i-1,j+1]==1:
            count-=300
    return count

def score_for_special_pos_wolf_corner_line_middle(matrix,action):
    score=0
    if (action[2]==4 and action[3]==4) or (action[2]==0 and action[3]==0) or (action[2]==0 and action[3]==4) or (action[2]==4 and action[3]==0):
        score+=check_how_many_empty_pos_around_wolf_corner(matrix=matrix,action=action)
    elif action[2]==4 or action[2]==0 or action[3]==0 or action[3]==4:
        score+=check_how_many_empty_pos_around_wolf_line(matrix=matrix,action=action)
    else:
        score+=check_how_many_around_wolf_middle(matrix=matrix,action=action)
    return score
#let wolf go towards the sheep
def find_sheep_positions(board):
    sheep_positions = []


    for row in range(5):
        for col in range(5):
            if board[row][col] == 1:
                sheep_positions.append((row, col))
    return sheep_positions

def distance(sheep, wolf):
    return abs(sheep[0] - wolf[0]) + abs(sheep[1] - wolf[1])

def find_closest_sheep(board,wolf_position):
    sheep_positions= find_sheep_positions(board)
    min_distance = float('inf')
    closest_sheep = None

    for sheep_position in sheep_positions:
        dist = distance(sheep_position, wolf_position)
        if dist < min_distance:
            min_distance = dist
            closest_sheep = sheep_position

    return closest_sheep, min_distance

def check_near_sheep(matrix,action):    
    currentpos_i=action[0]
    currentpos_j=action[1]
    count=0
    wolf_position_current=tuple([currentpos_i,currentpos_j])
    closest_sheep, min_distance=find_closest_sheep(board=matrix,wolf_position=wolf_position_current)
    wolf_position_next=tuple([action[2],action[3]])
    if distance(closest_sheep,wolf_position_next)<min_distance:
        count+=10
    return count

def check_if_wolf_original_in_danger_corner(matrix,action):
    i=action[0]
    j=action[1]
    count=0

    if (i-1<0 and j-1<0):       
        if matrix[i+1,j]==1 and matrix[i,j+1]==0:
            if matrix[i,j+2]==1:
                count+=10000
        if matrix[i,j+1]==1 and matrix[i+1,j]==0:
            if matrix[i+2,j]==1:
                count+=10000
    if i-1<0 and j+1==5:
        if matrix[i+1,j]==1 and matrix[i,j-1]==0:
            if matrix[i,j-2]==1:
                count+=10000
        if matrix[i,j-1]==1 and matrix[i+1,j]==0:
            if matrix[i+2,j]==1:
                count+=10000
    if i==4 and j==0:
        if matrix[i-1,j]==1 and matrix[i,j+1]==0:
            if matrix[i,j+2]==1:
                count+=10000
        if matrix[i,j+1]==1 and matrix[i-1,j]==0:
            if matrix[i-2,j]==1:
                count+=10000
    if i==4 and j==4:
        if matrix[i-1,j]==1 and matrix[i,j-1]==0:
            if matrix[i,j-2]==1:
                count+=10000
        if matrix[i,j-1]==1 and matrix[i-1,j]==0:
            if matrix[i-2,j]==1:
                count+=10000
    return count


def chech_if_wolf_original_in_danger_line(matrix,action):
    i=action[0]
    j=action[1]
    count=0
    if matrix[i-1,j]==1 and matrix[i,j+1]==1 and matrix[i+1,j]==1:
        if matrix[i-1,j-1]==1 or matrix[i+1,j-1]==1:
            count+=10000
    if matrix[i-1,j]==1 and matrix[i+1,j]==1 and matrix[i,j-1]==1:
        if matrix[i-1,j+1]==1 or matrix[i+1,j+1]==1:
            count+=10000
    if matrix[i,j+1]==1 and matrix[i+1,j]==1 and matrix[i,j-1]==1:
        if matrix[i-1,j-1]==1 or matrix[i-1,j+1]==1:
            count+=10000
    return count
def chech_if_wolf_original_in_danger_middle(matrix,action):
    i=action[0]
    j=action[1]
    count=0
    if matrix[i-1,j]==1 and matrix[i,j+1]==1 and matrix[i+1,j]==1:
        if matrix[i-1,j-1]==1 or matrix[i+1,j-1]==1:
            count+=10000
    if matrix[i-1,j]==1 and matrix[i+1,j]==1 and matrix[i,j-1]==1:
        if matrix[i-1,j+1]==1 or matrix[i+1,j+1]==1:
            count+=10000
    if matrix[i,j+1]==1 and matrix[i+1,j]==1 and matrix[i,j-1]==1:
        if matrix[i-1,j-1]==1 or matrix[i-1,j+1]==1:
            count+=10000
    return count
#check wolf orininal position,if he is in the special position, it must escape from the trap
def chech_original_position_wolf(matrix,action):
    score=0
    if (action[0]==4 and action[1]==4) or (action[0]==0 and action[1]==0) or (action[0]==0 and action[1]==4) or (action[0]==4 and action[1]==0):
        score+=check_if_wolf_original_in_danger_corner(matrix=matrix,action=action)
    elif action[0]==4 or action[1]==0 or action[0]==0 or action[1]==4:
        score+=chech_if_wolf_original_in_danger_line(matrix=matrix,action=action)
    else:
        score+=chech_if_wolf_original_in_danger_middle(matrix=matrix,action=action)
    return score

#check the remaining position around the wolf
def checkwolf_surround_remaining_cells(action,state):
    count=0
    i=action[2]
    j=action[3]
    #wolf in the middle
    if i+1<5 and i-1>=0 and j+1<5 and j-1>=0:
        if state[i-1][j]==0:
            count+=1
        if state[i-1][j]==0:
            count+=1
        if state[i][j-1]==0:
            count+=1
        if state[i][j+1]==0:
            count+=1
    #wolf in the corner
    #left up corner
    if i-1<0 and j-1<0:
        if state[i+1,j]==0:
            count+=1
        if state[i,j+1]==0:
            count+=1
    #right up corner
    if i-1<0 and j+1==5:
        if state[i+1,j]==0:
            count+=1
        if state[i,j-1]==0:
            count+=1
    #left bottom:
    if i==4 and j==0:
        if state[i-1,j]==0:
            count+=1
        if state[i,j+1]==0:
            count+=1
    #right bottom:
    if i==4 and j==4:
        if state[i-1,j]==0:
            count+=1
        if state[i,j-1]==0:
            count+=1
    #wolf in the line
    #bottom
    if i==4 and j!=0 and j!=4:
        if state[i,j-1]==0:
            count+=1
        if state[i-1,j]==0:
            count+=1
        if state[i,j+1]==0:
            count+=1
    #up
    if i==0 and j!=0 and j!=4:
        if state[i,j-1]==0:
            count+=1
        if state[i+1,j]==0:
            count+=1
        if state[i,j+1]==0:
            count+=1
    #left
    if j==0 and i!=0 and i!=4:
        if state[i-1,j]==0:
            count+=1
        if state[i,j+1]==0:
            count+=1
        if state[i+1,j]==0:
            count+=1
    #right
    if j==4 and i!=0 and i!=4:
        if state[i-1,j]==0:
            count+=1
        if state[i,j-1]==0:
            count+=1
        if state[i+1,j]==0:
            count+=1
           
    return count


def score_wolf_possible_actions(matrix):
    resultdic=dict()
    actions=getallpossibleactions_wolf(matrix=matrix)
    for a in actions:
        resultdic[a]=0
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==2:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        action=tuple([i,j,i+1,j])
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                       # resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        action=tuple([i,j,i-1,j])
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                       # resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if j+1<5:
                    if matrix[i,j+1]==0:
                        action=tuple([i,j,i,j+1])
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                        #resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        action=tuple([i,j,i,j-1])
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                       # resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if i+2<5:
                    if matrix[i+2,j]==1 and matrix[i+1,j]==0:
                        action=tuple([i,j,i+2,j])
                        resultdic[action]+=10000
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                       # resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if i-2>=0:
                    if matrix[i-2,j]==1 and matrix[i-1,j]==0:
                        action=tuple([i,j,i-2,j])
                        resultdic[action]+=10000
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                       # resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if j+2<5:
                    if matrix[i,j+2]==1 and matrix[i,j+1]==0:
                        action=tuple([i,j,i,j+2])
                        resultdic[action]+=10000
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                      #  resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
                if j-2>=0:
                    if matrix[i,j-2]==1 and matrix[i,j-1]==0:
                        action=tuple([i,j,i,j-2])
                        resultdic[action]+=10000
                        resultdic[action]+=score_for_special_pos_wolf_corner_line_middle(matrix=matrix,action=action)
                       # resultdic[action]+=chech_original_position_wolf(matrix=matrix,action=action)
                        num=checkwolf_surround_remaining_cells(action=action,state=matrix)
                        resultdic[action]+=num*2
                        resultdic[action]+=check_near_sheep(matrix=matrix,action=action)
    return resultdic

def getallpossibleactions_sheep(matrix):
    candidates=[]
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==1:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        candidates.append(tuple([i,j,i+1,j]))
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        candidates.append(tuple([i,j,i-1,j]))
                if j+1<5:
                    if matrix[i,j+1]==0:
                        candidates.append(tuple([i,j,i,j+1]))
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        candidates.append(tuple([i,j,i,j-1]))
    
    return candidates
#check if next step there is wolf around
def check_if_there_wolf_around_sheep_next_step(matrix,action):
    i=action[2]
    j=action[3]
    count=0
    if i+2<5:
        if matrix[i+2,j]==2:
            count-=10000
    if i-2>=0:
        if matrix[i-2,j]==2:
            count-=10000
    if j+2<5:
        if matrix[i,j+2]==2:
            count-=10000
    if j-2>=0:
        if matrix[i,j-2]==2:
            count-=10000
    return count

#check if there is wolf around the sheep this step
def check_if_there_wolf_around_sheep_this_step(matrix,action):
    i=action[0]
    j=action[1]
    if i+2<5:
        if matrix[i+2,j]==2:
            return True
    if i-2>=0:
        if matrix[i-2,j]==2:
            return True
    if j+2<5:
        if matrix[i,j+2]==2:
            return True
    if j-2>=0:
        if matrix[i,j-2]==2:
            return True
    return False

#check if next step can trap the wolf
def check_if_sheep_can_trap_wolf(matrix,action):
    i=action[2]
    j=action[3]
    count=0
    if i+1<5:
        if matrix[i+1,j]==2:
            count+=5
    if i-1>=0:
        if matrix[i-1,j]==2:
            count+=5
    if j-1>=0:
        if matrix[i,j-1]==2:
            count+=5
    if j+1<5:
        if matrix[i,j+1]==2:
            count+=5
    return count
#check now whether sheep is dangerous
def check_sheep_is_in_danger_now(matrix,action):
    i=action[0]
    j=action[1]
    count=0
    ifdanger=check_if_there_wolf_around_sheep_this_step(matrix=matrix,action=action)
    if ifdanger:
        count+=100
    return count
#score sheep
def score_sheep(matrix):
    resultdic=dict()
    actions=getallpossibleactions_sheep(matrix=matrix)
    for a in actions:
        resultdic[a]=0
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==1:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        action=tuple([i,j,i+1,j])
                        resultdic[action]+=check_if_there_wolf_around_sheep_next_step(matrix=matrix,action=action)
                        resultdic[action]+=check_sheep_is_in_danger_now(matrix=matrix,action=action)
                        resultdic[action]+=check_if_sheep_can_trap_wolf(matrix=matrix,action=action)
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        action=tuple([i,j,i-1,j])
                        resultdic[action]+=check_if_there_wolf_around_sheep_next_step(matrix=matrix,action=action)
                        resultdic[action]+=check_sheep_is_in_danger_now(matrix=matrix,action=action)
                        resultdic[action]+=check_if_sheep_can_trap_wolf(matrix=matrix,action=action)
                if j+1<5:
                    if matrix[i,j+1]==0:
                        action=tuple([i,j,i,j+1])
                        resultdic[action]+=check_if_there_wolf_around_sheep_next_step(matrix=matrix,action=action)
                        resultdic[action]+=check_sheep_is_in_danger_now(matrix=matrix,action=action)
                        resultdic[action]+=check_if_sheep_can_trap_wolf(matrix=matrix,action=action)
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        action=tuple([i,j,i,j-1])
                        resultdic[action]+=check_if_there_wolf_around_sheep_next_step(matrix=matrix,action=action)
                        resultdic[action]+=check_sheep_is_in_danger_now(matrix=matrix,action=action)
                        resultdic[action]+=check_if_sheep_can_trap_wolf(matrix=matrix,action=action)

    return resultdic

GLOB_TIMER = Timer()
def AIAlgorithm(filename, movemade): # a showcase for random walk
    iter_num=filename.split('/')[-1]
    iter_num=iter_num.split('.')[0]
    iter_num=int(iter_num.split('_')[1])
    matrix=load_matrix(filename)
    GLOB_TIMER.tic()
    if movemade==True:
        [start_row, start_col, end_row, end_col]=next_move_wolf(matrix)
        matrix2=copy.deepcopy(matrix)
        matrix2[end_row, end_col]=2
        matrix2[start_row, start_col]=0
            
    if movemade==False:
        [start_row, start_col, end_row, end_col]=next_move_sheep(matrix)
        matrix2=copy.deepcopy(matrix)
        matrix2[end_row, end_col]=1
        matrix2[start_row, start_col]=0
    duration = GLOB_TIMER.toc()
    print('[{}] move: {}, time:{}, cumulative time: {}'.format(iter_num+1, [start_row, start_col, end_row, end_col], duration, GLOB_TIMER.get_time()))
        
    matrix_file_name_output=filename.replace('state_'+str(iter_num), 'state_'+str(iter_num+1)) 
    write_matrix(matrix2, matrix_file_name_output)

    return start_row, start_col, end_row, end_col


