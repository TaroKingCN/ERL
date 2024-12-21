import os
import pickle
import random
import time

import gym
import numpy as np
import torch

from .core import mod_utils as utils
from .core.agent import Agent

from arch_gym.envs.DRAMEnv import DRAMEnv
from arch_gym.envs import dramsys_wrapper
from arch_gym.envs.envHelpers import helpers
import envlogger
from .parameters import Parameters


class PDERL():
    def __init__(self,
                 parameters: Parameters,
                 n_dim,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7):
        self.n_dim = n_dim
        parameters.lb = lb
        parameters.ub = ub
        parameters.precision = precision
        self.parameters = parameters
        print("pderl init")

    pass

    def run(self):
        print("pderl run")
        parameters = self.parameters
        tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
        frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
        time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')
        ddpg_tracker = utils.Tracker(parameters, ['ddpg'], '_score.csv')
        selection_tracker = utils.Tracker(parameters, ['elite', 'selected', 'discarded'], '_selection.csv')

        # Create Env
        env = dramsys_wrapper.make_dramsys_env(reward_formulation=parameters.reward_formulation)
        parameters.action_dim = self.n_dim
        parameters.state_dim = self.n_dim

        # Write the parameters to a the info file and print them
        parameters.write_params(stdout=True)

        # Seed
        torch.manual_seed(parameters.seed)
        np.random.seed(parameters.seed)
        random.seed(parameters.seed)

        # Create Agent
        agent = Agent(parameters, env)
        print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:',
              parameters.action_dim)

        next_save = parameters.next_save;
        time_start = time.time()
        while agent.num_frames <= parameters.num_frames:
            stats = agent.train()
            best_train_fitness = stats['best_train_fitness']
            erl_score = stats['test_score']
            elite_index = stats['elite_index']
            ddpg_reward = stats['ddpg_reward']
            print(stats['pg_loss'])
            policy_gradient_loss = stats['pg_loss']
            behaviour_cloning_loss = stats['bc_loss']
            population_novelty = stats['pop_novelty']

            print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
                  ' Train_Max:', '%.2f' % best_train_fitness if best_train_fitness is not None else None,
                  ' Test_Score:', '%.2f' % erl_score if erl_score is not None else None,
                  ' Avg:', '%.2f' % tracker.all_tracker[0][1],
                  ' ENV:  ' + parameters.env_name,
                  ' DDPG Reward:', '%.2f' % ddpg_reward,
                  ' PG Loss:', '%.4f' % policy_gradient_loss)
            print(agent.evolver.selection_stats['total'])

            elite = agent.evolver.selection_stats['elite'] / agent.evolver.selection_stats['total']
            selected = agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']
            discarded = agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']

            tracker.update([erl_score], agent.num_games)
            frame_tracker.update([erl_score], agent.num_frames)
            time_tracker.update([erl_score], time.time() - time_start)
            ddpg_tracker.update([ddpg_reward], agent.num_frames)
            selection_tracker.update([elite, selected, discarded], agent.num_frames)

            # Save Policy
            if agent.num_games > next_save:
                next_save += parameters.next_save
                if elite_index is not None:
                    torch.save(agent.pop[elite_index].actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                       'evo_net.pkl'))

                    if parameters.save_periodic:
                        save_folder = os.path.join(parameters.save_foldername, 'models')
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        actor_save_name = os.path.join(save_folder, 'evo_net_actor_{}.pkl'.format(next_save))
                        critic_save_name = os.path.join(save_folder, 'evo_net_critic_{}.pkl'.format(next_save))
                        buffer_save_name = os.path.join(save_folder, 'champion_buffer_{}.pkl'.format(next_save))

                        torch.save(agent.pop[elite_index].actor.state_dict(), actor_save_name)
                        torch.save(agent.rl_agent.critic.state_dict(), critic_save_name)
                        with open(buffer_save_name, 'wb+') as buffer_file:
                            pickle.dump(agent.rl_agent.buffer, buffer_file)

                print("Progress Saved")
