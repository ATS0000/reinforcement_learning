import numpy as np
import random

n = 4 #フィールド1辺のマス数
q_table = np.array([[ 0 for i in range(6)] for j in range(n*n*n)])

class Agent:
    # 行動a(t)を求める関数 -------------------------------------
    def get_action(next_state, episode, epsilon):
        '''
               #徐々に最適行動のみをとる、ε-greedy法
        epsilon = 10. * (1 / (episode + 1))
        '''
        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(q_table[next_state])
            '''
        p = 0.85
        if p >= np.random.uniform(0, 1):
            next_action = np.argmax(q_table[next_state[0]][next_state[1]])
            '''

        else:
            next_action = np.random.choice([0, 1, 2, 3, 4, 5])
        return next_action

    # 行動関数
    def act(action, state):
        reward = 0
        if action == 0:
            next_state = state + 1
        if action == 1:
            next_state = state - 4
        if action == 2:
            next_state = state - 1
        if action == 3:
            next_state = state + 4
        if action == 4:
            next_state = state + 16
        if action == 5:
            next_state = state - 16


        #壁に当たったらもとに戻る
        if state % n == 0 and action == 2:#左
            next_state = state
        if state % n == n-1 and action == 0:#右
            next_state = state
        if 0 <= state <= n*n-1 and action == 5:#手前
            next_state = state
        if n*n*(n-1) <= state <= n*n*n-1 and action == 4:#奥
            next_state = state
        for k in range(n):#上
            if n*n*(n-k-1) <= state <= n*n*(n-k-1)+n-1 and action == 1:
                next_state = state
            else:
                continue
        for k in range(n):#下
            if n*n*(n-k)-n <= state <= n*n*(n-k)-1 and action == 3:
                next_state = state
            else:
                continue


        #報酬を計算
        reward += 1 #移動したら報酬
        if next_state == n*n*n-1-n:
            #print('failure')
            reward = -50
            #break #穴に落ちたら原点
        if next_state == 42:
            #print('successfuly!')
            reward = 100 #ここではボーナスpointもらえる
        if next_state == n*n*n-1:
            #print('successfuly!')
            reward = 1000 #ここではボーナスpointもらえる
        if state == next_state:
            reward = 0 #移動しないなら報酬は0

        return next_state, reward

    # Qテーブルを更新する関数 -------------------------------------
    def update_Qtable(q_table, state, action, reward, next_state, alpha, gamma):
        #gamma = 0.95
        #alpha = 0.2
        next_Max_Q=max(q_table[next_state][0],
                q_table[next_state][1],
                q_table[next_state][2],
                q_table[next_state][3],
                q_table[next_state][4],
                q_table[next_state][5])
        q_table[state][action] = (1 - alpha) * q_table[state][action] +\
                alpha * (reward + gamma * next_Max_Q)
        return q_table

# stateの値をフィールド化する
class Henkan:

    def Henkan(state):
        x = 1 + (state % n)
        for i in range(n*n):
            if n*i <= state <= (i+1)*n - 1:
                y = i % n + 1
            else:
                continue
        for i in range(n):
            if (i+1)*(n*n-1)+i-n*n+1 <= state <= (i+1)*(n*n-1)+i:
                z = i + 1
            else:
                continue

        return x, y, z
