import numpy as np
import time 
import random


import torch
import argparse

from .parameters import Parameters
from .core.operator_runner import OperatorRunner

from arch_gym.envs.DRAMEnv import DRAMEnv
from arch_gym.envs import dramsys_wrapper
from arch_gym.envs.envHelpers import helpers


import os

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)







from .core import mod_utils as utils
from .core.agent import Agent

class ERLRE2():
    def __init__(self,
                 parameters: Parameters,
                 n_dim,
                 lb=-1, ub=1,
                 precision=1e-7):
        self.n_dim = n_dim
        parameters.lb = lb
        parameters.ub = ub
        parameters.action_dim = self.n_dim
        parameters.state_dim = self.n_dim
        parameters.precision = precision
        self.parameters = parameters
        print("ERLRE2 init")

    pass

    def run(self):
        parameters = self.parameters
        # Create Env
        env = dramsys_wrapper.make_dramsys_env(reward_formulation=parameters.reward)
        print("env.action_space.low", env.action_space.low, "env.action_space.high", env.action_space.high)

        # Seed
        os.environ['PYTHONHASHSEED'] = str(parameters.seed)
        env.seed(parameters.seed)
        torch.manual_seed(parameters.seed)
        np.random.seed(parameters.seed)
        random.seed(parameters.seed)

        tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
        frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
        time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')
        ddpg_tracker = utils.Tracker(parameters, ['ddpg'], '_score.csv')
        selection_tracker = utils.Tracker(parameters, ['elite', 'selected', 'discarded'], '_selection.csv')

        # Tests the variation operators after that is saved first with -save_periodic
        if parameters.test_operators:
            operator_runner = OperatorRunner(parameters, env)
            operator_runner.run()
            exit()


        # Create Agent
        agent = Agent(parameters, env)
        print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:',
              parameters.action_dim)

        next_save = parameters.next_save;
        time_start = time.time()
        while agent.num_frames <= parameters.num_frames:
        # while agent.num_frames <= 1:
            stats = agent.train()
            best_train_fitness = stats['best_train_fitness']
            erl_score = stats['test_score']
            elite_index = stats['elite_index']
            ddpg_reward = stats['ddpg_reward']
            policy_gradient_loss = stats['pg_loss']
            behaviour_cloning_loss = stats['bc_loss']
            population_novelty = stats['pop_novelty']
            current_q = stats['current_q']
            target_q = stats['target_q']
            pre_loss = stats['pre_loss']
            before_rewards = stats['before_rewards']
            add_rewards = stats['add_rewards']
            l1_before_after = stats['l1_before_after']
            keep_c_loss = stats['keep_c_loss']
            pvn_loss = stats['pvn_loss']
            min_fintess = stats['min_fintess']
            best_old_fitness = stats['best_old_fitness']
            new_fitness = stats['new_fitness']

            print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
                  ' Train_Max:', '%.2f' % best_train_fitness if best_train_fitness is not None else None,
                  ' Test_Score:', '%.2f' % erl_score if erl_score is not None else None,
                  ' Avg:', '%.2f' % tracker.all_tracker[0][1],
                  ' ENV:  ' + parameters.env_name,
                  ' DDPG Reward:', '%.2f' % ddpg_reward,
                  ' PG Loss:', '%.4f' % policy_gradient_loss)

            elite = agent.evolver.selection_stats['elite'] / agent.evolver.selection_stats['total']
            selected = agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']
            discarded = agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']

            print()

            min_fintess = stats['min_fintess']
            best_old_fitness = stats['best_old_fitness']
            new_fitness = stats['new_fitness']
            best_reward = np.max([ddpg_reward, erl_score])

            # parameters.wandb.log(
            #     {'best_reward': best_reward, 'add_rewards': add_rewards,
            #      'pvn_loss': pvn_loss, 'keep_c_loss': keep_c_loss, 'l1_before_after': l1_before_after,
            #      'pre_loss': pre_loss, 'num_frames': agent.num_frames, 'num_games': agent.num_games,
            #      'erl_score': erl_score, 'ddpg_reward': ddpg_reward, 'elite': elite, 'selected': selected,
            #      'discarded': discarded,
            #      'policy_gradient_loss': policy_gradient_loss, 'population_novelty': population_novelty,
            #      'best_train_fitness': best_train_fitness, 'behaviour_cloning_loss': behaviour_cloning_loss})














