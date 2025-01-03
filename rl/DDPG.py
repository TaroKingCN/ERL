from arch_gym.envs.DRAMEnv import DRAMEnv
from arch_gym.envs import dramsys_wrapper
from arch_gym.envs.envHelpers import helpers

from __future__ import division
from .parameters import Parameters
import gym
import os
import gc
import numpy as np
import time
import random
import torch
import actor
import buffer



class DDPG():
    def __init__(self,
                 parameters: Parameters,
                 env,
                 n_dim,
                 lb=-1, ub=1,
                 precision=1e-7):
        self.n_dim = n_dim
        parameters.lb = lb
        self.env = env
        parameters.ub = ub
        parameters.action_dim = self.n_dim
        parameters.state_dim = self.n_dim
        parameters.precision = precision
        self.parameters = parameters
        print("DDPG init")

    pass

    def action_trans(self, state, lb, ub):
        lb = np.array(lb)
        ub = np.array(ub)
        X = lb + (ub - lb + 1) * state
        X = np.where(X > ub, ub, X)
        X = np.where(X < lb, lb, X)
        return np.floor(X)

    def run(self):
        print("ddpg run")
        parameters = self.parameters

        # Create Env
        env = self.env
        parameters.action_dim = self.n_dim
        parameters.state_dim = self.n_dim

        # Write the parameters to a the info file and print them
        parameters.write_params(stdout=True)

        # Seed
        torch.manual_seed(parameters.seed)
        np.random.seed(parameters.seed)
        random.seed(parameters.seed)

        # buffer
        ram = buffer.MemoryBuffer(parameters.buffer_size)

        # Create Agent
        agent = actor.Actor(parameters, ram)
        print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:',
              parameters.action_dim)

        for _ep in range(parameters.episode):
            observation = env.reset()
            print('EPISODE :- ', _ep)
            rewards = []
            for r in range(parameters.step):
                state = np.float32(observation)

                action = agent.get_exploration_action(state)
                action = self.action_trans(action, parameters.lb, parameters.ub)
                _, reward, done, info = env.step(action)

                rewards.append(reward)

                if done:
                    new_state = None
                else:
                    new_state = np.float32(action)
                    # push this exp in ram
                    ram.add(state, action, reward, new_state)

                observation = action

                # perform optimization
                agent.optimize()
                if done:
                    break

            # check memory consumption and clear memory
            gc.collect()

            total_reward = 0;
            for reward in rewards:
                total_reward += reward

            print("avg_reward: ", total_reward/len(rewards))

            if _ep % 100 == 0:
                agent.save_models(_ep)










