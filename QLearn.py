import gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningExperiment:
    def __init__(self, alpha, gamma, epsilon, max_e, seed=0):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = max_e
        self.env = gym.make('LunarLander-v2')
        np.random.seed(seed)
        self.env.seed(seed)

    def bestAction(self, currS, q_val):
        actions = [q_val.get((currS,0), 0), q_val.get((currS, 1), 0), 
                       q_val.get((currS, 2), 0), q_val.get((currS,3), 0)]
        max_Q = max(actions)
        action = actions.index(max_Q)
        return action, max_Q

    def epsilon_greedy(self, currS, epsilon, q_val):
        randomNo = np.random.random()
        if randomNo < epsilon: # Explore
            action = np.random.randint(4)
        else: # Exploit
            action, _ = self.bestAction(currS,q_val)
        return action 

    def train(self):
        """Create the Q table"""
        np.random.seed(20240)
        all_R = []
        for i in range(self.episode):
            currS = self.env.reset()
            notDone = True
            curr_R = 0
            if i % 10000 == 0:
                print("Episode ", str(i))
            while notDone:
                currS = tuple(currS)
                action = self.epsilon_greedy(currS, self.epsilon, self.Q)
                nextS, reward, done, _ = self.env.step(action)
                curr_R += reward
                nextS = tuple(nextS)
                _, maxNextQ = self.bestAction(nextS, self.Q)
                self.Q[(currS, action)] = self.Q.get((currS, action), 0) + \
                self.alpha * (reward + self.gamma * maxNextQ - self.Q.get((currS, action), 0))
                currS = nextS
                notDone = (done != True)
                if done:
                    all_R.append(curr_R)
        self.plot(all_R, 'Training Q-Learning Rewards against Episode', 'Episodes', 'Rewards', './results/training_q_rew.png')
        
    def run(self):
        all_R = []
        for i in range(100):
            currS = self.env.reset()
            notDone = True
            curr_R = 0
            while notDone:
                currS = tuple(currS)
                action, _ = self.bestAction(currS, self.Q)
                nextS, reward, done, _ = self.env.step(action)
                curr_R += reward
                nextS = tuple(nextS)
                currS = nextS
                notDone = (done != True)
                if done:
                    all_R.append(curr_R)
        self.plot(all_R, 'Trained Q-Learning Rewards against Episode', 'Episodes', 'Rewards', './results/trained_q_rew.png')
        print(np.mean(np.array(all_R)))

    def plot(self, items, title, xaxis, yaxis, filename):
        plt.plot(range(len(items)), items)
        plt.title(title)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.savefig(filename)
        plt.close()
                