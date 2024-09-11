import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from hdqn_mdp import *
from my_env import *
from load import *
from hdqn import *
from my_env import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def hdqn_learning(
    agent,
    ):

    """The h-DQN learning algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    agent:
        a h-DQN agent consists of a meta-controller and controller.
    num_episodes:
        Number (can be divided by 1000) of episodes to run for. Ex: 12000
    exploration_schedule: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    gamma: float
        Discount Factor
    """
    ###############
    # RUN ENV     #
    ###############
    # Keep track of useful statistics

    env = StochasticMDPEnv()

    print("初始化完毕！")
    #循环episode
    for i_thousand_episode in range(100):
        batch_size1 = 128
        batch_size2 = 1

        #重置环境
        #(3164)
        current_state = env.reset()
        state_t = current_state.clone()
        # 外层meta-controller是否完成，即整个大网是否可控
        done = 0
        chooesed = []
        while not done:
            state_t = current_state.clone()
            # 选择目标，即选择子网，目标是让选中的子网中，被选中的节点数增加
            goal = agent.select_goal(state_t,0.8,env.subgraph_control,False)
            # 初始化外部奖励
            total_extrinsic_reward = 0

            #如果controller完成目标，或者整个子网已经可控
            goal_reached = False
            isdone = 0
            while not isdone and not goal_reached:

                #需要把整体的state，转为子图的sub-state
                #(50)
                sub_state = env.get_substate(goal).clone()
                sub_state_t = sub_state.unsqueeze(0)
                sub_state_t = sub_state_t.unsqueeze(2)
                # 根据目标选择动作，即子网中选择哪个节点
                if int(torch.sum(sub_state_t)) == agent.num_action :
                    isdone = 1
                    env.subgraph_control[goal] = 1
                    break
                else:
                    isdone = 0
                action = agent.select_action(sub_state_t,goal,0.8,False)

                chooesed.append(env.station_sub_action[goal][action])
                # 更新状态，并获取reward奖励
                # 存在子图k，action更新了，current_state没更新的情况
                # print(torch.sum(env.current_state),end=" ")
                next_state, extrinsic_reward, isdone = env.step(goal,action)
                # print(torch.sum(env.current_state))

                if extrinsic_reward > 0:
                    goal_reached = True


                sub_state_next = env.get_substate(goal).clone()
                sub_state_next_t = sub_state_next.unsqueeze(0)
                sub_state_next_t = sub_state_next_t.unsqueeze(2)

                # controller经验存放到经验池
                agent.ctrl_replay_memory.store_transition(sub_state_t, action,goal, extrinsic_reward, sub_state_next_t.clone(), isdone)
                # # 更新网络
                #
                if agent.ctrl_replay_memory.can_sample(batch_size2):
                    agent.update_controller()

                if agent.meta_replay_memory.can_sample(batch_size1):
                    agent.update_meta_controller()

                #计算外部reward
                total_extrinsic_reward += extrinsic_reward

                #更新状态
                current_state = env.current_state.clone()

                # print(torch.sum(current_state),extrinsic_reward,env.station_sub_action[goal][action],isdone,goal_reached)

            done = torch.tensor(env.meta_isdone()).reshape(1,1)
            if torch.sum(current_state) == 3164:
                done = 1
            # print(done)
            # meta-controller存放到经验池

            agent.meta_replay_memory.store_transition(state_t, goal, total_extrinsic_reward, current_state, done)

        print("第"+str(i_thousand_episode)+"轮：选了",len(set(chooesed)),"个节点:",chooesed[:5])


        if (i_thousand_episode+1) %20 == 0:
            torch.save(agent.meta_controller,"meta-N300-2-"+str(i_thousand_episode+1)+".pt")
            torch.save(agent.meta_replay_memory,"meta-memory-N300-2-"+str(i_thousand_episode+1)+".pt")
            # torch.save(chooesed,"chooesed_"+str(i_thousand_episode+1)+".pt")
            torch.save(agent.controller,"controller-N300-2-"+str(i_thousand_episode+1)+".pt")
            torch.save(agent.ctrl_replay_memory, "controller-memory-N300-2-" + str(i_thousand_episode + 1) + ".pt")



    return agent

if __name__ == '__main__':
    agent = hDQN(batch_size1=128,batch_size2=1)


    hdqn_learning(agent)
























