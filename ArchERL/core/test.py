import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Shared MLP (State Embedding Net)
class StateEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(StateEmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )

    def forward(self, state):
        return self.net(state)


# Individual Actor (MLP)
class Actor(nn.Module):
    def __init__(self, embedding_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state_embedding):
        return self.net(state_embedding)


# Critic Network
class Critic(nn.Module):
    def __init__(self, embedding_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state_embedding, action):
        x = torch.cat([state_embedding, action], dim=1)
        return self.net(x)


# Replay Buffer (for sampling experience)
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros(buffer_size)
        self.next_states = np.zeros((buffer_size, state_dim))
        self.dones = np.zeros(buffer_size)
        self.ptr, self.size = 0, 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )

#
# # Initialize networks
# state_dim = 10
# action_dim = 2
# embedding_dim = 16
# buffer_size = 100000
# batch_size = 64
#
# state_embedding_net = StateEmbeddingNet(state_dim, embedding_dim)
# actor = Actor(embedding_dim, action_dim)
# critic = Critic(embedding_dim, action_dim)
#
# target_state_embedding_net = StateEmbeddingNet(state_dim, embedding_dim)
# target_actor = Actor(embedding_dim, action_dim)
# target_critic = Critic(embedding_dim, action_dim)
#
# # Copy initial parameters to target networks
# target_state_embedding_net.load_state_dict(state_embedding_net.state_dict())
# target_actor.load_state_dict(actor.state_dict())
# target_critic.load_state_dict(critic.state_dict())
#
# # Optimizers
# actor_optimizer = optim.Adam(list(state_embedding_net.parameters()) + list(actor.parameters()), lr=1e-4)
# critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
#
# # Replay Buffer
# replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
#
# # Training loop
# for step in range(10000):
#     # Sample a batch
#     states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
#
#     # Forward pass
#     state_embeddings = state_embedding_net(states)
#     next_state_embeddings = target_state_embedding_net(next_states)
#
#     # Critic loss
#     with torch.no_grad():
#         target_actions = target_actor(next_state_embeddings)
#         target_q = target_critic(next_state_embeddings, target_actions)
#         target_value = rewards + (1 - dones) * 0.99 * target_q
#
#     q_values = critic(state_embeddings, actions)
#     critic_loss = ((q_values - target_value) ** 2).mean()
#
#     critic_optimizer.zero_grad()
#     critic_loss.backward()
#     critic_optimizer.step()
#
#     # Actor loss
#     predicted_actions = actor(state_embeddings)
#     actor_loss = -critic(state_embeddings, predicted_actions).mean()
#
#     actor_optimizer.zero_grad()
#     actor_loss.backward()
#     actor_optimizer.step()
#
#     # Target network updates
#     for target_param, param in zip(target_state_embedding_net.parameters(), state_embedding_net.parameters()):
#         target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
#     for target_param, param in zip(target_actor.parameters(), actor.parameters()):
#         target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
#     for target_param, param in zip(target_critic.parameters(), critic.parameters()):
#         target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
#

if __name__ == "__main__":

    # Initialize networks
    state_dim = 10
    action_dim = 2
    embedding_dim = 16
    buffer_size = 100000
    batch_size = 64

    state_embedding_net = StateEmbeddingNet(state_dim, embedding_dim)
    actor = Actor(embedding_dim, action_dim)

    for name, param in state_embedding_net.named_parameters():
        print(name)
        print(param)


    target_state_embedding_net = StateEmbeddingNet(state_dim, embedding_dim)
    target_actor = Actor(embedding_dim, action_dim)


    # Copy initial parameters to target networks
    target_state_embedding_net.load_state_dict(state_embedding_net.state_dict())
    target_actor.load_state_dict(actor.state_dict())

    initial_params = {name: param.clone() for name, param in target_state_embedding_net.named_parameters()}

    # Optimizers
    actor_optimizer = optim.Adam(list(state_embedding_net.parameters()) + list(actor.parameters()), lr=1e-1)

    # Define a simple loss function
    loss_fn = torch.nn.MSELoss()

    input_data = torch.rand((5, 10))  # Batch of 5, input_dim=10
    target_data_embedding = torch.rand((5, 16))
    target_data_action = torch.rand((5, 2))
    target_data_critic = torch.rand((5, 1))

    # Forward pass
    output = actor(state_embedding_net(input_data))
    loss = loss_fn(output, target_data_action)

    # Backward pass
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()

    for name, param in state_embedding_net.named_parameters():
        print(name)
        print(param)

