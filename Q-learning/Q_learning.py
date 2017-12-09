'''
Author: Daniel Eftekhari
Correspondences to daniel.eftekhari@mail.utoronto.ca

Q-learning (see README for details)

The program saves the initial and final mazes (.png and .txt).
The program saves the path that was taken just before rewards are changed,
or equivalently halfway through the number of iterations, and at program termination.
The program saves the most probable moves for each position in the maze just before rewards are changed,
or equivalently halfway through the number of iterations, and at program termination.
'''

from utils import *
import numpy as np
import time
import matplotlib.pyplot as plt

def forward_pass(plot_maze, maze, shape, prob):
    path = np.array([[int(shape[0]/2),0]], dtype=np.uint16)
    steps = 0
    while maze[path[-1,0], path[-1,1]] > 1:
        # show agent position in maze
        if plot_maze == True:
            show_position(np.copy(maze), path[-1], steps)

        counter = 0
        random_number = np.random.rand(1)
        if random_number < (prob[path[steps,0],path[steps,1],0] + counter) and random_number >= counter:
            new_step = np.array([[path[steps,0],path[steps,1]-1]])
            path = np.concatenate((path, new_step))
            steps += 1
            continue
        else:
            counter += prob[path[steps,0],path[steps,1],0]
        if random_number < (prob[path[steps,0],path[steps,1],1] + counter) and random_number >= counter:
            new_step = np.array([[path[steps,0]-1,path[steps,1]]])
            path = np.concatenate((path, new_step))
            steps += 1
            continue
        else:
            counter += prob[path[steps,0],path[steps,1],1]
        try:
            if random_number < (prob[path[steps,0],path[steps,1],2] + counter) and random_number >= counter:
                new_step = np.array([[path[steps,0],path[steps,1]+1]])
                path = np.concatenate((path, new_step))
                steps += 1
                continue
            else:
                counter += prob[path[steps,0],path[steps,1],2]
        except:
            pass
        try:
            if random_number < (prob[path[steps,0],path[steps,1],3] + counter) and random_number >= counter:
                new_step = np.array([[path[steps,0]+1,path[steps,1]]])
                path = np.concatenate((path, new_step))
                steps += 1
                continue
            else:
                counter += prob[path[steps,0],path[steps,1],3]
        except:
            pass

    return path, steps

def backward_pass(maze, Q, prob, path, steps, map_of_rewards, alpha, discount_steps, gain):
    counter = 0
    while steps > 0:
        # update Q
        Q = update_Q(Q, path, steps, map_of_rewards, alpha, discount_steps, counter)
        steps -= 1
        counter += 1

        # softmax
        prob = softmax(maze, Q, prob, path, steps, gain)

    return Q, prob

def update_Q(Q, path, steps, map_of_rewards, alpha, discount_steps, counter):
    if path[steps,0] == path[steps-1,0] and path[steps,1] == (path[steps-1,1] - 1):
        Q[path[steps-1,0],path[steps-1,1],0] = Q[path[steps-1,0],path[steps-1,1],0]+ alpha*(map_of_rewards[path[steps,0],path[steps,1]]+(discount_steps ** counter)*max(Q[path[steps,0],path[steps,1]])-Q[path[steps-1,0],path[steps-1,1],0])
    elif path[steps,0] == (path[steps-1,0] -1) and path[steps,1] == path[steps-1,1]:
        Q[path[steps-1,0],path[steps-1,1],1] = Q[path[steps-1,0],path[steps-1,1],1]+ alpha*(map_of_rewards[path[steps,0],path[steps,1]]+(discount_steps ** counter)*max(Q[path[steps,0],path[steps,1]])-Q[path[steps-1,0],path[steps-1,1],1])
    elif path[steps,0] == path[steps-1,0] and path[steps,1] == (path[steps-1,1] + 1):
        Q[path[steps-1,0],path[steps-1,1],2] = Q[path[steps-1,0],path[steps-1,1],2]+ alpha*(map_of_rewards[path[steps,0],path[steps,1]]+(discount_steps ** counter)*max(Q[path[steps,0],path[steps,1]])-Q[path[steps-1,0],path[steps-1,1],2])
    elif path[steps,0] == (path[steps-1,0] +1) and path[steps,1] == path[steps-1,1]:
        Q[path[steps-1,0],path[steps-1,1],3] = Q[path[steps-1,0],path[steps-1,1],3]+ alpha*(map_of_rewards[path[steps,0],path[steps,1]]+(discount_steps ** counter)*max(Q[path[steps,0],path[steps,1]])-Q[path[steps-1,0],path[steps-1,1],3])

    return Q

def softmax(maze, Q, prob, path, steps, gain):
    softmax_total = 0.0
    if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
        softmax_total += np.exp(gain * Q[path[steps-1,0],path[steps-1,1],0])
    if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
        softmax_total += np.exp(gain * Q[path[steps-1,0],path[steps-1,1],1])
    try:
        if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
            softmax_total += np.exp(gain * Q[path[steps-1,0],path[steps-1,1],2])
    except:
        pass
    try:
        if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
            softmax_total += np.exp(gain * Q[path[steps-1,0],path[steps-1,1],3])
    except:
        pass

    if path[steps-1,1] >= 1 and maze[path[steps-1,0],path[steps-1,1]-1] != 1:
        prob[path[steps-1,0],path[steps-1,1],0] = np.exp(gain * Q[path[steps-1,0],path[steps-1,1],0]) / softmax_total
    if path[steps-1,0] >= 1 and maze[path[steps-1,0]-1,path[steps-1,1]] != 1:
        prob[path[steps-1,0],path[steps-1,1],1] = np.exp(gain * Q[path[steps-1,0],path[steps-1,1],1]) / softmax_total
    try:
        if maze[path[steps-1,0],path[steps-1,1]+1] != 1:
            prob[path[steps-1,0],path[steps-1,1],2] = np.exp(gain * Q[path[steps-1,0],path[steps-1,1],2]) / softmax_total
    except:
        pass
    try:
        if maze[path[steps-1,0]+1,path[steps-1,1]] != 1:
            prob[path[steps-1,0],path[steps-1,1],3] = np.exp(gain * Q[path[steps-1,0],path[steps-1,1],3]) / softmax_total
    except:
        pass

    return prob

def main(load_maze, change_values, plot_maze, x_dim, y_dim, prob_not_wall, prob_reward, num_iterations, alpha, discount_steps, gain, gain_factor):
    '''

    :param load_maze: if True, load previously used maze, else create new maze
    :param change_values: if True, change reward values at num_iterations / 2, else maintain current values
    :param plot_maze: if True, display agent position in maze
    :param x_dim: Number of rows in the maze
    :param y_dim: Number of columns in the maze
    :param prob_not_wall: The probability that a block will not be a wall
    :param prob_reward: The probability that a non-wall block will be a reward
    :param num_iterations: Number of iterations the program runs for.
                           At num_iterations / 2, the gain is reset, and if change_values == True,
                           the program randomly changes the reward values
    :param alpha: Learning rate
    :param discount_steps: Discount factor
    :param gain: Initial trade-off value between exploration and exploitation
                            At num_iterations / 2, the program resets gain
    :param gain_factor: Multiplicative factor for gain
    :return: None

    '''

    # interactive feature of matplotlib
    if plot_maze == True:
        plt.ion()

    # initialize maze
    shape = [x_dim, y_dim]
    maze = create_maze(load_maze, shape, prob_not_wall, prob_reward)

    # map of rewards
    map_of_rewards = (maze < 1) * maze

    # initialize Q and probabilities
    Q = np.zeros((maze.shape[0], maze.shape[1], 4))
    prob = initialize_prob(maze, shape)

    # Begin experiment
    gain2 = gain
    for iterations in range(num_iterations):
        if iterations % 1000 == 0:
            print(iterations)
        if iterations == int(num_iterations/2-1):
            gain = gain2
            maze = update_maze(load_maze, change_values, maze, shape)
            map_of_rewards = (maze < 1) * maze

        # forward pass
        path, steps = forward_pass(plot_maze, maze, shape, prob)
        if iterations == (num_iterations-1) or iterations == int(num_iterations/2-1):
            save_policy(prob, iterations)
            save_path(path, iterations)

        # backward pass
        Q, prob = backward_pass(maze, Q, prob, path, steps, map_of_rewards, alpha, discount_steps, gain)
        gain *= gain_factor

if __name__ == "__main__":
    # parameters
    load_maze = True
    change_values = True
    plot_maze = False
    x_dim = 9
    y_dim = 6
    prob_not_wall = 0.75
    prob_reward = 0.12
    num_iterations = 18000
    alpha = 0.5
    discount_steps = 0.95
    # gain: high values -> exploit, low values -> explore
    gain = 0.05
    # multiplicative factor for the gain, applied after each iteration
    gain_factor = 1.0005

    # Stopwatch
    start_time = time.time()

    # Q-learning
    main(load_maze, change_values, plot_maze, x_dim, y_dim, prob_not_wall, prob_reward, num_iterations, alpha, discount_steps, gain, gain_factor)

    print("--- %s seconds ---" % (time.time() - start_time))