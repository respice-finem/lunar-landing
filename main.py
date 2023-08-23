from DQN import DQNExperiment
from QLearn import QLearningExperiment

# DQN Experiment 1 (Hidden Units)
def dqn_e1():
    # Fixed params
    max_epsilon = 1
    min_epsilon = 0.001
    e_decay = 0.996
    max_e = 2000
    disc_rate = 0.99
    alpha = 1e-4
    batch_size = 128

    # Variable Params
    hidden_units = [32, 64, 128, 256]
    all_r = []
    all_L = []
    all_S = []
    legends = []

    # Iterate through the hidden units
    for h in hidden_units:
        l1 = (8, h)
        l2 = (h, h)
        dqn = DQNExperiment(max_epsilon, min_epsilon, e_decay, max_e, disc_rate, alpha, batch_size, l1,l2)
        r, l, s = dqn.train()
        all_r.append(r)
        all_L.append(l)
        all_S.append(s)
        legends.append(f'Hidden Unit {h}')
        

    filename = f'e1_hidden_unit_experiment'
    dqn.plot(all_r, 'DQN Training Rewards against Episodes', 'Episodes','Rewards', f'./results/{filename}_train_rew', legends)
    dqn.plot(all_L, 'DQN Training Loss against Steps', 'Steps','Loss', f'./results/{filename}_train_loss', legends)
    dqn.plot(all_S, 'DQN Training Step against Episodes', 'Episodes','Step', f'./results/{filename}_train_step', legends)


# DQN Experiment 2 (Replay Batch Size)
def dqn_e2():
    
    # Fixed params
    max_epsilon = 1
    min_epsilon = 0.001
    e_decay = 0.996
    max_e = 2000
    disc_rate = 0.99
    alpha = 1e-4
    l1 = (8, 256)
    l2 = (256,256)

    # Variable Params
    batches = [32, 64, 128, 256]
    all_r = []
    all_L = []
    all_S = []
    legends = []
    
    # Iterate through the different batch sizes
    for b in batches:
        batch_size = b
        dqn = DQNExperiment(max_epsilon, min_epsilon, e_decay, max_e, disc_rate, alpha, batch_size, l1,l2)
        r, l, s = dqn.train()
        all_r.append(r)
        all_L.append(l)
        all_S.append(s)
        legends.append(f'Batch Size {b}')

    filename = f'e2_replay_batch_experiment'
    dqn.plot(all_r, 'DQN Training Rewards against Episodes', 'Episodes','Rewards', f'./results/{filename}_train_rew', legends)
    dqn.plot(all_L, 'DQN Training Loss against Steps', 'Steps','Loss', f'./results/{filename}_train_loss', legends)
    dqn.plot(all_S, 'DQN Training Step against Episodes', 'Episodes','Step', f'./results/{filename}_train_step', legends)

# DQN Experiment 3 (Discount Rate)
def dqn_e3():

    # Fixed params
    max_epsilon = 1
    min_epsilon = 0.001
    e_decay = 0.996
    max_e = 2000
    alpha = 1e-4
    batch_size = 128
    l1 = (8, 256)
    l2 = (256,256)


    # Variable Params
    disc_rates = [0.85, 0.9, 0.99, 0.999]
    all_r = []
    all_L = []
    all_S = []
    legends = []

    # Iterate through different discount rates
    for i in range(len(disc_rates)):
        disc_rate = disc_rates[i]
        dqn = DQNExperiment(max_epsilon, min_epsilon, e_decay, max_e, disc_rate, alpha, batch_size, l1,l2)
        r, l, s = dqn.train()
        all_r.append(r)
        all_L.append(l)
        all_S.append(s)
        legends.append(f'Disc Rate: {disc_rate}')

    filename = f'e3_disc_rate_experiment'
    dqn.plot(all_r, 'DQN Training Rewards against Episodes', 'Episodes','Rewards', f'./results/{filename}_train_rew', legends)
    dqn.plot(all_L, 'DQN Training Loss against Steps', 'Steps','Loss', f'./results/{filename}_train_loss', legends)
    dqn.plot(all_S, 'DQN Training Step against Episodes', 'Episodes','Step', f'./results/{filename}_train_step', legends)

## DQN Experiment 4 (Memory Capacity)
def dqn_e4():

     # Fixed params
    max_epsilon = 1
    min_epsilon = 0.001
    e_decay = 0.996
    max_e = 2000
    disc_rate = 0.99
    alpha = 1e-4
    batch_size = 128
    l1 = (8, 256)
    l2 = (256,256)

    # Variable Params
    memory_capacity = [1000, 10000, 100000, 1000000]
    all_r = []
    all_L = []
    all_S = []
    legends = []

    # Iterate through different memory capacity
    for cap in memory_capacity:
        dqn = DQNExperiment(max_epsilon, min_epsilon, e_decay, max_e, disc_rate, alpha, batch_size, l1,l2, seed=0, max_steps=500, max_mem=cap)
        r, l, s = dqn.train()
        all_r.append(r)
        all_L.append(l)
        all_S.append(s)
        legends.append(f'Capacity: {cap}')
    
    filename = f'e4_memory_capacity_experiment'
    dqn.plot(all_r, 'DQN Training Rewards against Episodes', 'Episodes','Rewards', f'./results/{filename}_train_rew', legends)
    dqn.plot(all_L, 'DQN Training Loss against Steps', 'Steps','Loss', f'./results/{filename}_train_loss', legends)
    dqn.plot(all_S, 'DQN Training Step against Episodes', 'Episodes','Step', f'./results/{filename}_train_step', legends)

# Q-Learning Final
def qlearn():
    alpha = 1e-4
    gamma = 0.999
    epsilon = 0.25
    max_e = 400000

    qlearn = QLearningExperiment(alpha, gamma, epsilon, max_e)
    qlearn.train()
    qlearn.run()

# DQN Best Model
def dqn_best():

    max_epsilon = 1
    min_epsilon = 0.001
    e_decay = 0.996
    max_e = 2000
    alpha = 1e-4
    hidden_units = [32, 64, 128, 256]
    batches = [32, 64, 128, 256]
    disc_rates = [0.85, 0.9, 0.99, 0.999]
    memory_capacity = [1000, 10000, 100000, 1000000]
    largest_rew = float('-inf')

    for h in hidden_units:
        for b in batches:
            for d in disc_rates:
                for m in memory_capacity:
                    l1 = (8, h)
                    l2 = (h,h)
                    batch_size = b
                    disc_rate = d
                    cap = m
                    legends = [f'Hidden {h}, Batches: {b}, Disc Rate: {d}, Capacity: {m}']
                    print(l1,l2,batch_size, disc_rate, cap)
                    dqn = DQNExperiment(max_epsilon, min_epsilon, e_decay, max_e, disc_rate, alpha, batch_size, l1,l2, seed=0, max_steps=500, max_mem=cap)
                    _, _, _ = dqn.train()
                    average_rew = dqn.run(f'./results/h_{h}_b_{b}_d_{d}_m_{m}_test.png', legends)
                    print(average_rew)
                    if average_rew > largest_rew:
                        print(f'Hidden {h}, Batches: {b}, Disc Rate: {d}, Capacity: {m}')
                        largest_rew = average_rew


# Main function
def main():

    # print("Running DQN Experiment 1 (Hidden Units)")
    # dqn_e1()
    # print("\n Running DQN Experiment 2 (Replay Batch Size)")
    # dqn_e2()
    # print("\n Running DQN Experiment 3 (Discount Rate)")
    # dqn_e3()
    # print("\n Running DQN Experiment 4 (Memory Capacity)")
    # dqn_e4()
    print("\n Q-Learning Training and Testing")
    qlearn()
    # print("\n DQN Search for Best Model")
    # dqn_best()

# Run main function
main()