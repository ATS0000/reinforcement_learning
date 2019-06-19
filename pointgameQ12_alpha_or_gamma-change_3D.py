#数ターンでどれだけポイントを獲得できるかのゲーム
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
from agent_Q12 import *
import datetime

n = 4
num_episodes = 500 #総試行回数
agent = Agent
henkan = Henkan
w = 10

#グラフ(キャンパス)を準備
fig = plt.figure()
#ax = fig.add_subplot(1,2,1, projection='3d')

graph = fig.add_subplot(1,1,1)
graph.set_title('Learning Process')
graph.set_xlabel('Episode')
graph.set_ylabel('Total Points')

'''
#フィールド(グラフ上)に報酬を置く
x1,y1,z1 = henkan.Henkan(n*n*n-1-n)
ax.scatter(x1,y1,z1, c='black',s=500, alpha=0.3) #穴をプロット
x2,y2,z2 = henkan.Henkan(42)
ax.scatter(x2,y2,z2, c='yellow',s=500, alpha=0.3) #宝をプロット
x3,y3,z3 = henkan.Henkan(n*n*n-1)
ax.scatter(x3,y3,z3, c='yellow',s=500, alpha=0.3) #宝をプロット
'''

start = datetime.datetime.now()
print('学習開始時刻', start)


for g in range(w):
#メインループ
    p = np.array([ 0 for i in range(num_episodes)]) #ポイント合計を入れる
    for episode in range(num_episodes):
        #fig = plt.figure()
        alpha = 0.2
        gamma = g*0.1
        #徐々に最適行動のみをとる、ε-greedy法
        epsilon = 50.* (1 / (episode + 1))
        state = 0 #スタート位置
        ims = []
        points = 0
        reward = 0
        action = np.argmax(q_table[state])

        #1ゲーム
        for i in range(50):
            next_state, reward = agent.act(action, state) #行動
            points += reward
            q_table = agent.update_Qtable(q_table, state, action, reward, next_state, alpha, gamma)
            #print('state = ',state, 'action = ',action, 'reward =',reward, 'points = ',points, 'next_state = ',next_state)
            action = agent.get_action(next_state, episode, epsilon) #次の行動aを求める
            #print(q_table,"\n")
            '''
            if episode == num_episodes - 1: # 最後の学習結果をグラフにする
                x, y, z = henkan.Henkan(state)
                im = ax.scatter(x,y,z, c='red', alpha=0.4)
                ims.append([im])
            '''
            state = next_state

        '''
        if points > num_episodes/5*100:
            print('complete!')
            break
        '''
        print(episode, 'points =' ,points)
        p[episode] = points

        #graph.scatter(episode, points,c='b',s=1)


    print('学習時間', datetime.datetime.now() - start)

    #ani = animation.ArtistAnimation(fig, ims, interval=300,repeat_delay=10)
    #ims[0].save('test.gif', save_all=True, append_ims=ims[1:], optimize=False, duration=40, loop=0)


    '''
    for i in range(20):
        x2 = i + 60
        y2 = 80
        plt.scatter(x2,y2, c='yellow',s=300) #宝をプロット

    for i in range(20):
        x3 = i + 60
        y3 = 60
        plt.scatter(x3,y3, c='black',s=500) #穴をプロット
    '''
    '''
    plt.grid(True)
    plt.show()
    time.sleep(0.1)

    ani.save('test.gif')
    '''


        #グラフをs個で移動平均
    s = 30
    b = np.ones(s)/s
    p_ave = np.convolve(p,b,'valid')

    plt.plot(p_ave, label = '{:.2g}'.format(g*0.1))
    plt.legend() # 凡例を表示

plt.show()


#from IPython.display import HTML
#HTML(ani.to_html5_video())
