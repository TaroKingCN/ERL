import pprint
import torch
import os


class Parameters:
    def __init__(self, cla, init=True):

        if not init:
            return
        cla = cla.parse_args()

        # Set the device to run on CUDA or CPU
        if  torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpfu')

        # ddpg
        self.gamma = 0.99
        self.tau = 0.001
        self.seed = 114514
        self.batch_size = 128
        self.use_done_mask = True
        self.buffer_size = 1000000
        self.ls = 128
        self.episode = cla.episode
        self.step = cla.step


        # ========================================== NeuroEvolution Params =============================================

        # Num of trials
        self.num_evals = 1

        # Save Results
        self.state_dim = None  # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.save_foldername = './log'

        self.n_dim = None

        self.lb = None
        self.ub = None
        self.precision = None
        self.summary_dir = cla.summary_dir
        self.reward_formulation = cla.reward_formulation
        self.workload = cla.workload

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=True):
        # Dump all the hyper-parameters in a file.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.save_foldername, 'info.txt'), 'a') as f:
            f.write(params)
