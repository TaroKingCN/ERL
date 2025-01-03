from arch_gym.envs.DRAMEnv import DRAMEnv
from arch_gym.envs import dramsys_wrapper
from arch_gym.envs.envHelpers import helpers

from .parameters import Parameters
import gym
import os
import gc
import numpy as np
import pandas as pd
import time
import random
import torch
from . import actor
from . import buffer



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

    def generate_run_directories(self):
        # Construct the exp name from seed and num_iter
        exp_name = str(self.parameters.workload) + "_step_" + str(self.parameters.step) + "_episode_" + str(self.parameters.episode)
        traject_dir = os.path.join(self.parameters.summary_dir, "ddpg_logs", self.parameters.reward_formulation, exp_name)
        # log directories for storing exp csvs
        exp_log_dir = os.path.join(self.parameters.summary_dir,"ddpg_logs",self.parameters.reward_formulation, exp_name)
        if not os.path.exists(exp_log_dir):
            os.makedirs(exp_log_dir)
        return traject_dir, exp_log_dir

    def log_fitness_to_csv(self, filename, fitness_dict):
        df = pd.DataFrame([fitness_dict['reward']])
        csvfile = os.path.join(filename, "fitness.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

        # append to csv
        df = pd.DataFrame([fitness_dict])
        csvfile = os.path.join(filename, "trajectory.csv")
        df.to_csv(csvfile, index=False, header=False, mode='a')

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

        dram_helper = helpers()
        print('Running DRAM', ' State_dim:', parameters.state_dim, ' Action_dim:',
              parameters.action_dim)

        fitness_hist={}
        traject_dir, exp_log_dir = self.generate_run_directories()
        for _ep in range(parameters.episode):
            observation = env.reset().observation
            print(observation)
            print('EPISODE :- ', _ep)
            rewards = []
            for r in range(parameters.step):
                state = np.float32(observation)

                action = agent.get_exploration_action(state)
                action_trans = self.action_trans(action, parameters.lb, parameters.ub)
                action_dic = dram_helper.action_decoder_ga(action_trans)
                _, reward, done, info = env.step(action_dic)

                rewards.append(reward)

                
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

            fitness_hist['iteration'] = _ep+1
            fitness_hist['reward'] = reward
            self.log_fitness_to_csv(exp_log_dir, fitness_hist)





