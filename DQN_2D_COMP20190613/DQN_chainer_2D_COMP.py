
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import myenv_2D_COMP
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import datetime

from chainer import training
from chainer.training import extensions

today = datetime.datetime.today()

#gym,envから必要なものを持ってくる---------------
env = gym.make('myenv_2D-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)
obs = env.reset()
#env.render()
print('initial observation:', obs)
action = env.action_space.sample()
t = 0
obs, reward, done, info, hole, trea, pos = env.step(action, t)
print('next observation:', obs)
print('reward:', reward)
print('done:', done)
print('info:', info)
#-----------------------------------------------

n = 4

#グラフ(キャンパス)を準備
fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(1,2,1)

graph = fig.add_subplot(1,2,2)
graph.set_title('Learning Process')
graph.set_xlabel('Episode')
graph.set_ylabel('Total Points')


n_episodes = 300

class RnadomActor:
    def __init__(self):
        self.i = 0
    def random_action_func(self):
        self.i = np.random.choice([0, 1, 2, 3])
        return self.i

ra = RnadomActor()

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = n*n
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)
'''
_q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_layers=2, n_hidden_channels=50)
'''

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.999

# Use epsilon-greedy for exploration
#explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    #1.0, 0.01, 0.001, random_action_func=env.action_space.sample)

explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.4, decay_steps=n_episodes, random_action_func=ra.random_action_func)


# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
#replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10 ** 6)


# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, phi=phi)

turn_num = 30
#--------------------------------
'''
batchsize = 4
trainy = np.array(([0],[1],[1],[1]), dtype=np.int32)
train = chainer.datasets.TupleDataset(obs, trainy)
train_iter = chainer.iterators.SerialIterator(train, batchsize)
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (n_episodes, 'n_episodes'))
trainer.extend(extensions.dump_graph('20181127'))
'''
#---------------------------------
start = datetime.datetime.now()
print('学習開始時刻', start)
#メインループ
for episode in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    P = 0  # return (sum of rewards)
    t = 1  # time step
    ims = []

    while not done and t < turn_num + 1:
        # Uncomment to watch the behaviour
        # env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _, hole, trea, pos = env.step(action, t)
        P += reward
        t += 1

        if episode == n_episodes: # 最後の学習結果をグラフにする
            x = pos[0]
            y = pos[1]
            im = ax.scatter(x,y, c='red',s=200, alpha=0.4)
            ims.append([im])
            #trea,holeが動くとき------------------
            '''
            x1 = hole[0]
            y1 = hole[1]
            im = plt.scatter(x1,y1, c='black',s=300, alpha=0.2)
            ims.append([im]) #穴をプロット
            x2 = trea[0]
            y2 = trea[1]
            im = plt.scatter(x2,y2, c='yellow',s=500, alpha=0.3) #宝をプロット
            ims.append([im]) #宝をプロット
            '''
            #--------------------------------------

        #print('obs:', obs,'action', action, 'P:', P, 'statistics:', agent.get_statistics())

    agent.stop_episode_and_train(obs, reward, done)

    print('EPISODE', episode,'P', P)

    graph.scatter(episode, P,s=1, c='b')

print('Finished.')
print('学習時間', datetime.datetime.now() - start)
#グラフプロット------------------------------

#フィールド(グラフ上)に報酬を置く
#trea,holeが静止のとき--------------------
x1 = hole[0]
y1 = hole[1]
ax.scatter(x1,y1, c='black',s=300, alpha=0.2)
x2 = trea[0]
y2 = trea[1]
ax.scatter(x2,y2, c='yellow',s=500, alpha=0.3) #宝をプロット
#----------------------------------------

ani = animation.ArtistAnimation(fig, ims, interval=100,repeat_delay=1000)
ani.save(today.strftime("%Y%m%d_%H-%M-%S") + 'data.gif')
plt.grid(True)
plt.show()
time.sleep(0.1)

#Save an agent to the 'agent' directory
#agent.save('agent_n3')

#---------------------------------------
# Uncomment to load an agent from the 'agent' directory
#agent.load('agent_n4')


#テストセッション--------------------------
for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < turn_num:
        #env.render()
        action = agent.act(obs)
        obs, reward, done, _, hole, trea, pos = env.step(action, t)
        R += reward
        t += 1
        #print('obs:', obs,'action', action, 'R:', R, 'statistics:', agent.get_statistics())
    print('test episode:', i, 'R:', R)
    agent.stop_episode()
