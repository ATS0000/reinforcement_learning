import sys

import gym
import numpy as np
import gym.spaces
import random

n = 4
class MyEnv(gym.Env):
    map = np.array([[ 0 for i in range(4)] for j in range(4)])
    #map = np.arange(n*n).reshape(n,n)
    MAX_STEPS = 20000

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # 東西南北
        self.observation_space = gym.spaces.Box(
            low=0,
            high=4,
            shape=(n,n)
        )
        self.reward_range = [-100., 100.]
        self._reset()

    def _reset(self):
        # 諸々の変数を初期化する
        self.pos = np.array([0, 0])
        self.ex_hole = np.array([n, n-1]) #穴の初期位置
        self.ex_trea = np.array([n, n]) #宝の初期位置
        self.steps = 0
        self.done = False
        return self._observe()

    def _step(self, action, t):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        if action == 0:
            next_pos = self.pos + [0, 1]
        elif action == 1:
            next_pos = self.pos + [0, -1]
        elif action == 2:
            next_pos = self.pos + [1, 0]
        elif action == 3:
            next_pos = self.pos + [-1, 0]

        if self._is_movable(next_pos):
            reward, hole, trea = self._get_reward(next_pos, t)
        else:
            next_pos = self.pos
            hole = self.ex_hole
            trea = self.ex_trea
            reward = -10 #移動しないなら報酬は0

        observation = self._observe()
        self.pos = next_pos
        self.done = self._is_done()
        self.ex_hole = hole
        self.ex_trea = trea
        return observation, reward, self.done, {}, hole, trea, self.pos

    def _get_reward(self, next_pos, t):
        reward = -1 #移動したら報酬

        if t % 10000 == 0: #every k turns ,change hole spot and treasure spot
            hole = np.array([random.randint(0,n-1),random.randint(0,n-1)])
            trea = np.array([random.randint(0,n-1),random.randint(0,n-1)])
        else:
            hole = self.ex_hole
            trea = self.ex_trea

        if (next_pos == hole).all():
            #print('failure')
            reward -= 5
            #break #穴に落ちたら減点
        '''
        u = 5
        if u <= next_pos[0] < self.MAP.shape[0] -u and u <= next_pos[1] < self.MAP.shape[1] -u:
            reward -= 100
        '''
        if (next_pos == trea).all():
            #print('successfuly!')
            reward += 100 #ここではボーナスpointもらえる

        return reward, hole, trea

    def _is_movable(self, next_pos):
        # マップの中にいるか
        return (
            0 <= next_pos[0] < self.map.shape[0]
            and 0 <= next_pos[1] < self.map.shape[1]
        )

    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            return True
        else:
            return False

    def _observe(self):
        # マップに勇者の位置を重ねて返す
        observation = self.map.copy()
        observation[tuple(self.pos)] = 9
        return observation

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

...
