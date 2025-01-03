import pprint
import torch
import os


class Parameters:
    def __init__(self, cla, init=True):

        if not init:
            return
        cla = cla.parse_args()

        # Set the device to run on CUDA or CPU
        if not cla.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpfu')

        # Render episodes
        self.render = cla.render
        self.env_name = cla.env
        self.save_periodic = cla.save_periodic

        self.num_frames = 10000

        # Novelty Search
        self.ns = cla.novelty
        self.ns_epochs = 10

        # DDPG params
        self.gamma = 0.99
        self.tau = 0.001
        self.seed = cla.seed
        self.batch_size = 128
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.buffer_size = 1000000
        self.ls = 128
        self.episode = 1000
        self.step = 10

        # Prioritised Experience Replay
        self.per = cla.per
        self.replace_old = True
        self.alpha = 0.7
        self.beta_zero = 0.5
        self.learn_start = (1 + self.buffer_size / self.batch_size) * 2
        self.total_steps = self.num_frames

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

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=True):
        # Dump all the hyper-parameters in a file.
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        with open(os.path.join(self.save_foldername, 'info.txt'), 'a') as f:
            f.write(params)