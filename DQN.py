import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Vanilla DQN without target network
class DQN(nn.Module):

    def __init__(self, l1, l2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(l1[0], l1[1])
        self.fc2 = nn.Linear(l2[0], l2[1])
        self.output = nn.Linear(l2[1], 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

# DQN Experiment
class DQNExperiment:

    def __init__(self, max_epsilon, min_epsilon, e_decay, max_e, disc_rate, alpha, batch_size, l1,l2, seed=0, max_steps=500, max_mem=10000):
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = e_decay
        self.max_e = max_e # Number of Episodes
        self.disc_rate = disc_rate # gamma
        self.alpha = alpha # Learning rate for optimizer
        self.batch_size = batch_size # Batch Size
        self.network = DQN(l1,l2)
        self.memory = deque([],maxlen=max_mem)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha) # Using ADAM as it has fast convergence
        self.env = gym.make('LunarLander-v2')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env._max_episode_steps = max_steps # Maximum Steps in Episode before stopping
        self.env.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
    def select_action(self, curr_state, epsilon):
        threshold = random.random()
        if threshold > epsilon: # Exploit
            with torch.no_grad():
                curr_state = torch.tensor(curr_state).view(1,-1)
                action = self.network(curr_state.to(self.device)).argmax().item()
        else: # Explore actions between 0 to 3 inclusive
            action = random.randint(0,3)
        return action
    
    # Gradient Descent
    def gdesc(self):
        # Get batch, each batch element is [state, action, next_state, reward, done]
        # Converted list of numpy arrays to numpy 2D arrays before changing to Tensor to speed up processing
        batch = random.sample(self.memory, self.batch_size)
        curr_state = torch.tensor(np.array([x[0] for x in batch]))
        next_state = torch.tensor(np.array([x[2] for x in batch]))
        action = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.int64).view(-1,1)
        reward = torch.tensor(np.array([x[3] for x in batch])).float().view(-1,1)
        done = 1 - torch.tensor(np.array([x[4] for x in batch])).float().view(-1,1)

        # Obtain y_j
        max_q = self.network(next_state.to(self.device)).max(1)[0].detach().view(-1,1) * done # If non-terminal only reward will be used
        target = torch.add(reward, max_q, alpha=self.disc_rate)
        predicted = self.network(curr_state.to(self.device)).gather(1, action).float()
        
        # Loss function
        loss = nn.MSELoss()
        output = loss(predicted, target)
        
        # Backprop
        self.optimizer.zero_grad()
        output.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return output.item() # Return loss
        
    # Main Experiment function
    def train(self):
        self.network.train()
        all_r = []
        all_S = []
        all_L = []
        moving_r = deque([], maxlen=100)
        total_e = 0
        # Play the game n times
        while total_e < self.max_e:
            state = self.env.reset()
            steps = 0
            curr_r = 0
            if total_e % 100 == 0:
                print("Episode ", total_e, " Average Reward: ", str(np.mean(np.array(moving_r))))
            # Playing the game
            while True:
                self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
                action = self.select_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                curr_r += reward
                self.memory.append([state, action, next_state, reward, done])
                state = next_state
                if len(self.memory) >= self.batch_size:
                    loss = self.gdesc()
                    all_L.append(loss)
                steps += 1
                if done:
                    all_r.append(curr_r)
                    all_S.append(steps)
                    moving_r.append(curr_r)
                    total_e += 1
                    break
            
            # If moving average >= 200, we stop training and save the weights
            if np.mean(np.array(moving_r)) >= 200:
                print("Episode ", total_e, " Reward: ", str(np.mean(np.array(moving_r))))
                torch.save(self.network.state_dict(), "./ll_dqn_params.pt")
                break
        if total_e == self.max_e:
            print("Episode ", total_e, " Reward: ", str(np.mean(np.array(moving_r))))
        return all_r, all_L, all_S
                    
    # Run the experiments after training the model
    def run(self, filename):
        self.network.eval()
        all_r = []
        all_S = []
        # Play the game 100 times
        for i in range(100):
            state = self.env.reset()
            steps = 0
            curr_r = 0
            # Playing the game
            while True:
                action = self.select_action(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                curr_r += reward
                steps += 1
                if done:
                    all_r.append(curr_r)
                    all_S.append(steps)
                    break
        self.plot(all_r, 'Trained DQN Rewards against Episodes', 'Episodes','Rewards', filename)
        return np.mean(np.array(all_r)) # Return average rewards
    
    # Plot graphs
    def plot(self, items, title, xaxis, yaxis, filename, legends):
        for i in range(len(items)):
            plt.plot(range(len(items[i])), items[i], label=legends[i])
        plt.title(title)
        plt.legend()
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.savefig(filename)
        plt.close()